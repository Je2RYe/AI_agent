#!/usr/bin/env python
import sys
import os
import warnings
from dotenv import load_dotenv
import streamlit as st

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, Process, Task
import time
import threading
import pandas as pd
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crew import VideoSummary
from manager import AdaptiveManager
from evaluation import compute_objective_metrics, compute_answer_grounding
from rag_engine import split_for_map_reduce, RecursiveChunker, HybridRetriever
from config import MAP_REDUCE_THRESHOLD_TOKENS, RAG_CONFIG
import db as db_module
from db import insert_chat_log, get_conn as db_get_conn
from judge_llm import run_judge_async

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv()


def _safe_tok(usage, key, default=0):
    """Safely extract a token field from either a dict or a UsageMetrics object."""
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default) or default
    return getattr(usage, key, default) or default

if 'manager' not in st.session_state:
    st.session_state.manager = AdaptiveManager()

manager = st.session_state.manager

def run_summarization(inputs, done_flag, model_name):
    """
    运行 Crew 的工作线程函数。
    支持两种路由（DESIGN §15.4）：
      - 全量直通：transcript < MAP_REDUCE_THRESHOLD_TOKENS（绝大多数视频）
      - Map-Reduce：transcript >= MAP_REDUCE_THRESHOLD_TOKENS（~60 分钟以上超长视频）
    RAG 已从摘要生成路径移除；仅保留于 Chat QA（正确用武之地）。
    """
    video_summary_crew = VideoSummary(model_name=model_name)
    start_time = time.time()
    
    done_flag['success'] = False
    done_flag['report_path'] = ""
    done_flag['transcript_path'] = ""
    done_flag['token_usage'] = {}
    done_flag['rag_info'] = None       # 保留字段（历史兼容）
    done_flag['map_reduce_info'] = None  # Map-Reduce 信息（若触发）

    try:
        # ========== Phase 1: Transcription ==========
        transcription_crew = video_summary_crew.create_transcription_crew()
        transcript_result = transcription_crew.kickoff(inputs=inputs)
        transcription_text = str(transcript_result)

        # ========== Phase 2: 路由决策 + Summarization ==========
        # DESIGN §15.4：RAG for QA，全量 for Summary
        #   < MAP_REDUCE_THRESHOLD_TOKENS  → 直接全量输入（最准确，召回率最高）
        #   >= MAP_REDUCE_THRESHOLD_TOKENS → Map-Reduce 分段汇总（工业界标准）
        rag_info = None
        tok1 = getattr(transcript_result, 'token_usage', None)
        est_tokens = int(len(transcription_text.split()) * 1.3)

        if est_tokens >= MAP_REDUCE_THRESHOLD_TOKENS:
            # ── Map-Reduce 路径（超长视频，~60 分钟以上）──
            segments = split_for_map_reduce(transcription_text)
            print(f"🗺️ Map-Reduce: transcript ~{est_tokens} tokens → {len(segments)} segments")

            seg_prompt_tok = seg_completion_tok = seg_total_tok = 0
            segment_summaries: list = []
            for i, seg in enumerate(segments, 1):
                print(f"  ✏️  Summarizing segment {i}/{len(segments)} ...")
                seg_crew = video_summary_crew.create_summary_crew_with_context(seg)
                seg_result = seg_crew.kickoff()
                segment_summaries.append(
                    f"## Segment {i} of {len(segments)}\n\n{seg_result}"
                )
                seg_tok = getattr(seg_result, 'token_usage', None)
                seg_prompt_tok     += _safe_tok(seg_tok, 'prompt_tokens')
                seg_completion_tok += _safe_tok(seg_tok, 'completion_tokens')
                seg_total_tok      += _safe_tok(seg_tok, 'total_tokens')

            # Merge step: 汇总各段摘要 → 最终摘要
            combined = "\n\n---\n\n".join(segment_summaries)
            merge_context = (
                "The following are partial summaries from different segments of a very long video. "
                "Please synthesize them into a single comprehensive structured summary.\n\n"
                + combined
            )
            print(f"  🔗 Merging {len(segments)} segment summaries ...")
            merge_crew = video_summary_crew.create_summary_crew_with_context(merge_context)
            result = merge_crew.kickoff()

            merge_tok = getattr(result, 'token_usage', None)
            merged_tokens = {
                'prompt_tokens':     _safe_tok(tok1, 'prompt_tokens')     + seg_prompt_tok     + _safe_tok(merge_tok, 'prompt_tokens'),
                'completion_tokens': _safe_tok(tok1, 'completion_tokens') + seg_completion_tok + _safe_tok(merge_tok, 'completion_tokens'),
                'total_tokens':      _safe_tok(tok1, 'total_tokens')      + seg_total_tok      + _safe_tok(merge_tok, 'total_tokens'),
            }
            done_flag['token_usage'] = merged_tokens
            print(f"✅ Map-Reduce complete: {len(segments)} segments merged")
            done_flag['map_reduce_info'] = {'n_segments': len(segments), 'est_tokens': est_tokens}

        else:
            # ── 全量直通路径（绝大多数视频）──
            print(f"⚡ Direct full-input (~{est_tokens} tokens), using direct summarization")
            summary_crew = video_summary_crew.create_summary_crew_with_context(transcription_text)
            result = summary_crew.kickoff()

            tok2 = getattr(result, 'token_usage', None)
            merged_tokens = {
                'prompt_tokens':     _safe_tok(tok1, 'prompt_tokens')     + _safe_tok(tok2, 'prompt_tokens'),
                'completion_tokens': _safe_tok(tok1, 'completion_tokens') + _safe_tok(tok2, 'completion_tokens'),
                'total_tokens':      _safe_tok(tok1, 'total_tokens')      + _safe_tok(tok2, 'total_tokens'),
            }
            done_flag['token_usage'] = merged_tokens

        generated_text = str(result)
        done_flag['rag_info'] = rag_info

        # ========== Phase 3: 文件保存 ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = "history_reports"
        os.makedirs(history_dir, exist_ok=True)
        
        with open("Video_Summary.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)

        clean_model_name = model_name.replace("/", "-").replace(":", "")
        transcript_filename = f"transcript_{timestamp}_{clean_model_name}.txt"
        saved_filename = f"summary_{timestamp}_{clean_model_name}.txt"
        saved_path = os.path.join(history_dir, saved_filename)
        transcript_path = os.path.join(history_dir, transcript_filename)

        with open(saved_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)

        done_flag['report_path'] = saved_path
        done_flag['transcript_path'] = transcript_path
        done_flag['success'] = True
            
    except Exception as e:
        done_flag['error'] = str(e)
        done_flag['success'] = False

    end_time = time.time()
    done_flag['latency'] = end_time - start_time
    done_flag['finished'] = True

def show_progress_placebo(inputs, dark_mode):
    st.title("🚀 Generating Summary")
    
    # --- [Step A: Analyze] 让 Manager 决定模型 ---
    selected_model = manager.decide_model()
    st.sidebar.info(f"🧠 Adaptive Core: Selected **{selected_model}** based on history.")
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # [Step B: Monitor] 初始化监控标志
    done_flag = {
        'finished': False,
        'latency': 0,
        'success': False,
        'error': None,
        'report_path': '',
        'transcript_path': '',
        'token_usage': {},
    }
    
    # 启动线程，传入选定的模型
    thread = threading.Thread(
        target=run_summarization,
        args=(inputs, done_flag, selected_model)
    )
    thread.start()

    progress = 0.0
    text_color = "#ffffff" if dark_mode else "#000000"
    success_color = "#66ff66" if dark_mode else "#006400"

    # 模拟进度条
    while progress < 0.99 and not done_flag['finished']:
        progress += 0.01
        time.sleep(0.2) 
        progress_bar.progress(min(progress, 0.99))
        progress_text.markdown(
            f"<p style='color: {text_color}; font-weight: bold;'>Progress: {int(progress * 100)}%</p>",
            unsafe_allow_html=True
        )

    # 等待线程真正结束
    thread.join()

    if done_flag['success']:
        progress_bar.progress(1.0)
        progress_text.markdown(
            f"<p style='color: {text_color}; font-weight: bold;'>Progress: 100%</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color: {success_color}; font-weight: bold;'>🎉 Task Completed in {done_flag['latency']:.2f}s!</p>",
            unsafe_allow_html=True
        )
        
        # 计算 Cost
        raw_usage = done_flag.get('token_usage', {})
        prompt_tokens = _safe_tok(raw_usage, 'prompt_tokens')
        completion_tokens = _safe_tok(raw_usage, 'completion_tokens')
        total_tokens = _safe_tok(raw_usage, 'total_tokens')

        cost = manager.calculate_cost(selected_model, prompt_tokens, completion_tokens)
        st.sidebar.success(f"💰 Cost: ${cost:.5f}")

        # === Objective metrics (consistency / completeness / coherence) ===
        objective_metrics = {
            'factual_consistency': None,
            'completeness': None,
            'coherence': None,
        }
        try:
            summary_text = ""
            transcript_text = ""
            if done_flag.get('report_path') and os.path.exists(done_flag['report_path']):
                with open(done_flag['report_path'], 'r', encoding='utf-8') as f:
                    summary_text = f.read()
            if done_flag.get('transcript_path') and os.path.exists(done_flag['transcript_path']):
                with open(done_flag['transcript_path'], 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                # Store for Chat RAG (lazy index built on first question)
                st.session_state['transcript_text'] = transcript_text
                st.session_state['rag_retriever'] = None  # Reset: invalidate stale index

            if summary_text and transcript_text:
                objective_metrics = compute_objective_metrics(summary_text, transcript_text)
            else:
                print(f"Skipping metrics: summary={len(summary_text)} chars, transcript={len(transcript_text)} chars")
        except Exception as e:
            print(f"Objective metrics failed: {e}")
            import traceback; traceback.print_exc()
            st.sidebar.warning(f"⚠️ Metrics computation failed: {e}")

        st.session_state['objective_metrics'] = objective_metrics

        # --- 初始化行为信号 ---
        st.session_state['behavior_signals'] = {
            'chat_depth': 0,
            're_asks': 0,
            'copy_clicks': 0,
            'expand_clicks': 0,
            'time_spent_sec': 0.0,
            'session_start_time': time.time(),
        }

        st.sidebar.subheader("Objective Metrics")
        fc = objective_metrics.get('factual_consistency')
        cp = objective_metrics.get('completeness')
        ch = objective_metrics.get('coherence')
        st.sidebar.metric("Consistency", f"{fc:.2f}" if fc is not None else "N/A")
        st.sidebar.metric("Completeness", f"{cp:.2f}" if cp is not None else "N/A")
        st.sidebar.metric("Coherence", f"{ch:.2f}" if ch is not None else "N/A")

        ws = manager.calculate_weighted_score(fc, cp, ch)
        if ws is not None:
            st.sidebar.metric("Weighted Score", f"{ws:.3f}")

        # --- 保存本次运行数据 ---
        report_path = done_flag.get('report_path', '')
        transcript_path = done_flag.get('transcript_path', '')

        st.session_state['last_run_data'] = {
            'model': selected_model,
            'latency': done_flag['latency'],
            'tokens': total_tokens,
            'cost': cost,
            'report_path': report_path,
            'transcript_path': transcript_path,
        }

        # --- 每次自动记录日志（无奇偶轮） ---
        # 读取 summary 用于预览
        summary_text_for_preview = ""
        try:
            if done_flag.get('report_path') and os.path.exists(done_flag['report_path']):
                with open(done_flag['report_path'], 'r', encoding='utf-8') as f:
                    summary_text_for_preview = f.read()
        except Exception:
            pass

        report_id = manager.log_execution(
            model_used=selected_model,
            latency=done_flag['latency'],
            total_tokens=total_tokens,
            cost=cost,
            report_path=report_path,
            transcript_path=transcript_path,
            factual_consistency=fc,
            completeness=cp,
            coherence=ch,
            summary_preview=summary_text_for_preview[:200],
            rag_info=done_flag.get('rag_info'),
        )
        st.session_state['last_report_id'] = report_id

        # --- Phase 2B: 异步 Judge-LLM 副轨（非阻塞） ---
        if report_id and summary_text_for_preview:
            _transcript_for_judge = st.session_state.get('transcript_text', '')
            if _transcript_for_judge:
                run_judge_async(
                    report_id=report_id,
                    summary=summary_text_for_preview,
                    transcript=_transcript_for_judge,
                )
                st.sidebar.caption("🧑‍⚖️ Judge-LLM running in background...")

        # --- 摘要生成路径状态显示 ---
        map_reduce_info = done_flag.get('map_reduce_info')
        if map_reduce_info:
            st.sidebar.info(
                f"🗺️ Map-Reduce: {map_reduce_info['n_segments']} segments ← "
                f"~{map_reduce_info['est_tokens']:,} tokens"
            )
        else:
            st.sidebar.info("⚡ Direct full-input (summary used full transcript)")

    else:
        st.error(f"Task Failed: {done_flag.get('error')}")
        if os.path.exists("Video_Summary.txt"):
            os.remove("Video_Summary.txt")

def set_theme(dark_mode: bool):
    if dark_mode:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg, chat_bg = (
            "#1f2430", "#121521", "#e0e6f1", "#232a3a", "#e0e6f1", "#2a3245", "#f0f3ff", "#3b4660", "#2a2f3d"
        )
    else:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg, chat_bg = (
            "#e6e8eb", "#ffffff", "#1c1c1c", "#f7f8fa", "#1c1c1c", "#f0f0f3", "#333333", "#d6d6d9", "#f2f2f2"
        )

    st.markdown(f"""
        <style>
        /* === Global Background and Text === */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]  {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}

        /* === Sidebar Styling === */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}

        /* === Selectbox (Dropdown) === */
        .stSelectbox div[data-baseweb="select"] > div:first-child {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
         
        }}

        /* === Main Content Area === */
        .main, .main * {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
           
        }}

        [data-testid="stTextInput"] input {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            padding: 0.5rem !important;
            font-size: 1rem !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
        }}

        /* === File Uploader === */
        [data-testid="stFileUploader"] > div > div,
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
        }}


        /* === Buttons === */
        button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: none !important;
            padding: 0.6rem 1.2rem;
            border-radius: 8px !important;
            font-weight: 600;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}
        button:hover {{
            background-color: {button_hover_bg} !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
            cursor: pointer;
        }}

        /* === Chat Messages === */
        [data-testid="stChatMessage"] {{
            background-color: {chat_bg} !important;
            border-radius: 12px !important;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        [data-testid="stChatMessageContent"] {{
            color: {text_color} !important;
        }}

        /* === Optional: Make chat input blend better === */
        div[data-testid="stChatInput"] {{
            border-top: 1px solid rgba(255,255,255,0.05);
            color: {text_color} !important;
        }}
        div[data-testid="stChatInput"] * {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
  
        }}
        div[data-testid="stChatInput"] button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: 1px solid #666 !important;
            border-radius: 0 12px 12px 0 !important;
        }}
        </style>
    """, unsafe_allow_html=True)



def main():
    # --- Session ID (stable within one browser session) ---
    if 'session_id' not in st.session_state:
        import uuid as _uuid
        st.session_state['session_id'] = str(_uuid.uuid4())[:8]

    # --- Language Selection ---
    language = st.sidebar.selectbox("🌐 Language / Langue", ["English", "Français"])

    translations = {
        "English": {
            "dark_mode": "Dark Mode",
            "title": "Content Summarizer",
            "youtube_input": "Enter YouTube URL:",
            "summarize_url": "Summarize Video from URL",
            "upload_file": "Upload an audio file (MP3, WAV, M4A)",
            "summarize_file": "Summarize from File",
            "warning_url": "Please enter a YouTube URL.",
            "warning_file": "Please upload an audio file.",
            "summary_title": "Generated Summary",
            "chat_title": "Chat About The Summary",
            "chat_input": "Ask a question about the summary...",
            "summary_info": "Your generated summary and chat will appear here once you provide a URL or file."
        },
        "Français": {
            "dark_mode": "Mode Sombre",
            "title": "Résumeur de Contenu",
            "youtube_input": "Entrez l'URL YouTube :",
            "summarize_url": "Résumer la vidéo à partir de l'URL",
            "upload_file": "Téléversez un fichier audio (MP3, WAV, M4A)",
            "summarize_file": "Résumer à partir du fichier",
            "warning_url": "Veuillez entrer une URL YouTube.",
            "warning_file": "Veuillez téléverser un fichier audio.",
            "summary_title": "Résumé Généré",
            "chat_title": "Discuter du Résumé",
            "chat_input": "Posez une question sur le résumé...",
            "summary_info": "Votre résumé généré et le chat apparaîtront ici une fois que vous aurez fourni une URL ou un fichier."
        }
    }

    t = translations[language]
    # Persist dark mode toggle across reruns
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    st.session_state.dark_mode = st.sidebar.checkbox(t["dark_mode"], value=st.session_state.dark_mode)
    dark_mode = st.session_state.dark_mode

    # Set theme based on stored dark mode state
    set_theme(dark_mode)

    st.sidebar.title(t["title"])

    # === 🌟 新增：资源实时监控仪表盘 ===
    st.sidebar.subheader("🔋 Resource Monitor")
    
    # 获取当前状态
    remaining_percent, used_cost, limit_cost = manager.get_resource_status()
    
    # 动态颜色：资源充足(绿) -> 警告(黄) -> 耗尽(红)
    bar_color = "green"
    if remaining_percent < 0.5: bar_color = "orange"
    if remaining_percent < 0.1: bar_color = "red"
    
    # 显示进度条
    st.sidebar.progress(remaining_percent)
    
    # 显示具体数值
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Budget Left", f"{remaining_percent*100:.0f}%")
    with col2:
        st.metric("Used / Limit", f"${used_cost:.2f}")
        #st.metric("Used / Limit", f"${used_cost:.0f} / ${limit_cost:.0f}")

    # 如果资源耗尽，显示警告
    if remaining_percent <= 0.02:
        st.sidebar.error("🚨 BANKRUPTCY: System locked to Lite model!")

    with st.sidebar.expander("⚙️ System Control (Admin)", expanded=False):
        st.caption("Monitor & Override MAPE-K Loop")
        
        # 1. 模式选择：自动自适应 vs 手动强制
        mode = st.radio("Adaptation Mode", ["Auto (MAPE-K)", "Manual Override"])
        
        selected_model_override = None
        if mode == "Manual Override":
            # 读取 manager 中定义的模型列表
            from config import MODEL_TIERS
            selected_model_override = st.selectbox("Force Model:", MODEL_TIERS)
        
        # 2. Chat RAG 开关
        chat_rag_enabled = st.checkbox(
            "💬 RAG for Chat Q&A", value=True,
            help="Uses Hybrid Search (SBERT + BM25 + RRF) to retrieve relevant transcript passages when you ask questions. "
                 "Note: RAG is NOT used for summary generation — summaries always use full transcript input for maximum recall."
        )
        st.session_state['chat_rag_enabled'] = chat_rag_enabled

        # 3. (可选) 显示当前系统状态
        if st.button("View Metrics CSV"):
            if os.path.exists("system_metrics.csv"):
                try:
                    df = pd.read_csv("system_metrics.csv")
                    st.dataframe(df.tail(5)) # 显示最近5条
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
            else:
                st.write("No history yet.")

        st.markdown("---")
        from config import COST_LIMIT_USD as _limit
        st.caption(f"💰 Budget cap: **${_limit:.2f}** — edit `COST_LIMIT_USD` in config.py to raise it.")
        if st.button("🔄 Reset Budget Counter", help="Clears cost accumulation in system_metrics.csv so Budget Left returns to 100%. Historical data is NOT deleted."):
            try:
                _df = pd.read_csv("system_metrics.csv")
                if not _df.empty:
                    _df["cost_usd"] = 0.0
                    _df.to_csv("system_metrics.csv", index=False)
                st.success("✅ Budget counter reset! Budget Left is now 100%.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

    youtube_url = st.sidebar.text_input(t["youtube_input"])
    if st.sidebar.button(t["summarize_url"]):
        if youtube_url:
            inputs = {'content': youtube_url}
            show_progress_placebo(inputs, dark_mode)
        else:
            st.sidebar.warning(t["warning_url"])

    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(t["upload_file"], type=["mp3", "wav", "m4a"])
    if st.sidebar.button(t["summarize_file"]):
        if uploaded_file is not None:
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            inputs = {'content': file_path}
            show_progress_placebo(inputs, dark_mode)
            os.remove(file_path)
        else:
            st.sidebar.warning(t["warning_file"])
    
    st.sidebar.markdown("---")
    # 🌟 新增：自适应系统测试区域 🌟
    st.sidebar.subheader("🛠️ Model Test (Internal)")
    
    # 定义一个函数，用于测试并显示结果
    def run_model_test(model_name: str, test_inputs: dict):
        st.sidebar.markdown(f"**Testing:** `{model_name}`...")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 实例化 Crew，传入指定的模型
            video_summary_crew = VideoSummary(model_name=model_name)
            # 运行 Crew，kickoff 需要 inputs
            # 注意：这里会执行完整的转录和总结流程
            result = video_summary_crew.create_summarization_crew().kickoff(inputs=test_inputs)
            
            latency = time.time() - start_time
            st.sidebar.success(f"✅ {model_name} Success: Latency={latency:.2f}s")
            
            # 清理生成的 summary file，避免影响主应用
            summary_file_path = "Video_Summary.txt"
            if os.path.exists(summary_file_path):
                os.remove(summary_file_path)

        except Exception as e:
            st.sidebar.error(f"❌ {model_name} Failed: {e}")
    
    # 假设一个固定的测试输入，例如一个短视频 URL (你需要替换成一个有效的短视频 URL)
    TEST_URL = "https://www.youtube.com/watch?v=uEHu8LIZUKI" 
    test_inputs = {'content': TEST_URL}

    # 按钮：测试 Flash 模型
    if st.sidebar.button("Test Gemini 2.0 Flash (Fast)"):
        # 清除之前的消息
        st.session_state.test_log = []
        run_model_test("gemini-2.0-flash-lite-001", test_inputs)

    # 按钮：测试 Pro 模型
    if st.sidebar.button("Test Gemini 2.5 pro (Smart)"):
        run_model_test("gemini-2.5-pro", test_inputs)

    st.sidebar.markdown("---")

    # ================================================================
    # MAIN AREA — Three Tabs: Summary | Reports | Metrics
    # ================================================================
    tab_summary, tab_reports, tab_metrics = st.tabs([
        "📄 " + t["summary_title"],
        "📂 Summary Reports",
        "📊 Objective Metrics",
    ])

    # ── Tab 1: Generated Summary (existing behavior) ──
    with tab_summary:
        summary_file_path = "Video_Summary.txt"

        if os.path.exists("Video_Summary.txt"):
            with open("Video_Summary.txt", "r", encoding="utf-8") as f:
                summary_content = f.read()

            # --- Restore transcript_text for Chat RAG if session was reloaded ---
            # transcript_text is only set during show_progress_placebo; if the user
            # opens the app with an existing Video_Summary.txt (from a prior session),
            # we need to recover it from DuckDB's most recent report transcript_path.
            if not st.session_state.get('transcript_text'):
                try:
                    from db import get_conn as _gc, get_reports as _gr
                    _conn = _gc()
                    _rep_df = _gr(_conn, include_deleted=False)
                    _conn.close()
                    if not _rep_df.empty:
                        _latest = _rep_df.iloc[0]
                        _tp = _latest.get('transcript_path', '')
                        if _tp and os.path.exists(str(_tp)):
                            with open(str(_tp), 'r', encoding='utf-8') as _f:
                                st.session_state['transcript_text'] = _f.read()
                            st.session_state['rag_retriever'] = None  # will be built on first Q
                            st.session_state['last_report_id'] = str(_latest.get('report_id', ''))
                except Exception as _re:
                    print(f"transcript restore failed: {_re}")

            st.subheader(t["summary_title"])
            st.markdown(summary_content)

            st.markdown("---")

            # === Chat Section ===
            st.subheader(t["chat_title"])

            # RAG status badge
            _has_transcript = bool(st.session_state.get('transcript_text'))
            _rag_on = st.session_state.get('chat_rag_enabled', True)
            if _has_transcript and _rag_on:
                st.caption("📚 RAG active — answers grounded in original transcript")
            elif not _has_transcript:
                st.caption("⚠️ No transcript loaded — RAG disabled (regenerate summary to enable)")
            else:
                st.caption("⚡ RAG off — answers based on summary only")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input(t["chat_input"]):
                # --- 行为信号追踪 ---
                if 'behavior_signals' not in st.session_state:
                    st.session_state['behavior_signals'] = {
                        'chat_depth': 0, 're_asks': 0,
                        'copy_clicks': 0, 'expand_clicks': 0,
                        'time_spent_sec': 0.0, 'session_start_time': time.time(),
                    }
                st.session_state['behavior_signals']['chat_depth'] += 1

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # --- RAG: retrieve transcript context for accurate Q&A ---
                        rag_context = ""
                        top_chunks_with_scores = []
                        rag_used_this_turn = False

                        if st.session_state.get('chat_rag_enabled', True):
                            transcript_text_chat = st.session_state.get('transcript_text', '')
                            if transcript_text_chat:
                                retriever = st.session_state.get('rag_retriever')
                                if retriever is None:
                                    with st.spinner("🔍 Indexing transcript for Q&A (one-time setup)..."):
                                        chunker = RecursiveChunker()
                                        chunks = chunker.chunk_transcript(transcript_text_chat)
                                        retriever = HybridRetriever(chunks)
                                        st.session_state['rag_retriever'] = retriever
                                top_chunks_with_scores = retriever.retrieve(prompt, top_k=RAG_CONFIG["top_k_retrieval"])
                                rag_context = "\n\n---\n\n".join(c.text for c, _ in top_chunks_with_scores)
                                rag_used_this_turn = bool(top_chunks_with_scores)

                        video_summary_crew = VideoSummary()
                        inputs = {
                            'summary': summary_content,
                            'transcript_context': rag_context,
                            'user_message': prompt
                        }
                        chat_crew = video_summary_crew.create_chat_crew()
                        response = chat_crew.kickoff(inputs=inputs)
                        response_str = str(response)

                        # --- RAG 可观测性指标计算 ---
                        retrieval_confidence = None
                        answer_grounding = None
                        n_chunks = len(top_chunks_with_scores)

                        if rag_used_this_turn and top_chunks_with_scores:
                            # 检索置信度: top-K RRF 分均値
                            rrf_scores = [score for _, score in top_chunks_with_scores]
                            retrieval_confidence = float(sum(rrf_scores) / len(rrf_scores))

                            # 答案锁定分: SBERT cos_sim(answer, retrieved_context)
                            try:
                                answer_grounding = compute_answer_grounding(response_str, rag_context)
                            except Exception as _ge:
                                print(f"Grounding score failed: {_ge}")

                        # --- 写入 chat_logs ---
                        try:
                            _conn = db_get_conn()
                            insert_chat_log(
                                conn=_conn,
                                question=prompt,
                                answer=response_str,
                                rag_used=rag_used_this_turn,
                                n_chunks_retrieved=n_chunks,
                                retrieval_confidence=retrieval_confidence,
                                answer_grounding_score=answer_grounding,
                                session_id=st.session_state.get('session_id', ''),
                                report_id=st.session_state.get('last_report_id', ''),
                                model_used=st.session_state.get('last_run_data', {}).get('model', ''),
                            )
                        except Exception as _le:
                            print(f"chat_log insert failed: {_le}")

                        st.markdown(response_str)

                        # --- 内联 RAG 诊断芯片 ---
                        if rag_used_this_turn:
                            conf_str  = f"{retrieval_confidence:.3f}" if retrieval_confidence  is not None else "N/A"
                            grnd_str  = f"{answer_grounding:.3f}"    if answer_grounding       is not None else "N/A"
                            grnd_icon = (
                                "✅" if answer_grounding is not None and answer_grounding >= 0.65 else
                                "⚠️" if answer_grounding is not None and answer_grounding >= 0.45 else
                                "❌" if answer_grounding is not None else "❓"
                            )
                            with st.expander(
                                f"📚 RAG | {n_chunks} chunks | Confidence {conf_str} | Grounding {grnd_str} {grnd_icon}",
                                expanded=False
                            ):
                                st.caption(
                                    """
**Retrieval Confidence** = avg RRF score of top-K chunks — “检索到了多少相关内容”  
**Answer Grounding** = SBERT sim(answer, retrieved context) — “回答有多少来自原文”

| 组合 | 含义 |
|---|---|
| 高置信 + 高锁定 | RAG 完整有效 ✅ |
| 高置信 + 低锁定 | 检索到但 LLM 未使用 ⚠️ |
| 低置信 + 任意 | 问题不适合 RAG 使用场景 ℹ️ |
                                    """
                                )
                                for i, (chunk, score) in enumerate(top_chunks_with_scores, 1):
                                    st.markdown(
                                        f"**Chunk {i}** (RRF={score:.4f}, pos={chunk.metadata.get('position_pct','?'):.0%})\n\n"
                                        f"> {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}"
                                    )

                st.session_state.messages.append({"role": "assistant", "content": response_str})

            # === 可选反馈面板（副轨 — DESIGN §5.3） ===
            st.markdown("---")
            with st.expander("📋 Optional Feedback (for offline validation)", expanded=False):
                with st.form("feedback_form"):
                    fb_score = st.slider("Rate this summary quality (1-10)", 1, 10, 5)
                    fb_text = st.text_area("Additional comments (optional)", "")
                    fb_action = st.radio("Was this summary helpful?", ["helpful", "neutral", "not_helpful"], index=1)
                    fb_submitted = st.form_submit_button("Submit Feedback")

                    if fb_submitted:
                        run_data = st.session_state.get('last_run_data', {})
                        manager.log_feedback(
                            model_used=run_data.get('model', ''),
                            report_path=run_data.get('report_path', ''),
                            user_feedback_score=fb_score,
                            user_feedback_text=fb_text,
                            objective_metrics=st.session_state.get('objective_metrics', {}),
                            user_action=fb_action,
                        )
                        st.success("Thank you for your feedback!")

        else:
            st.info(t["summary_info"])

    # ── Tab 2: Summary Reports (browse / soft-delete) ──
    with tab_reports:
        _render_reports_tab()

    # ── Tab 3: Objective Metrics (eval history + trends) ──
    with tab_metrics:
        _render_metrics_tab()


# ================================================================
# Reports Tab
# ================================================================
def _render_reports_tab():
    """📂 Summary Reports — 浏览历史报告、查看详情、软删除。"""
    from db import get_conn, get_reports, soft_delete_report, restore_report

    st.subheader("📂 Summary Reports")

    show_deleted = st.checkbox("Show deleted reports", value=False)

    try:
        conn = get_conn()
        reports_df = get_reports(conn, include_deleted=show_deleted)
        conn.close()
    except Exception as e:
        st.error(f"Could not load reports: {e}")
        return

    if reports_df.empty:
        st.info("No reports yet. Generate a summary to see it here.")
        return

    st.caption(f"Total: {len(reports_df)} report(s)")

    for idx, row in reports_df.iterrows():
        rid = row['report_id']
        model = row['model_used']
        created = str(row['created_at'])[:19]
        preview = row.get('summary_preview', '')[:120]
        is_del = row.get('is_deleted', False)

        # Header line
        status_icon = "🗑️" if is_del else "📄"
        header = f"{status_icon} **{model}** | {created}"

        with st.expander(header, expanded=False):
            st.text(f"Report ID: {rid}")
            st.text(f"Cost: ${row.get('cost_usd', 0):.5f} | Tokens: {row.get('total_tokens', 0)} | Latency: {row.get('latency_sec', 0):.1f}s")

            if preview:
                st.markdown(f"**Preview:** {preview}...")

            # Full report button
            rpath = row.get('report_path', '')
            if rpath and os.path.exists(rpath):
                with open(rpath, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                with st.expander("📖 Full Report", expanded=False):
                    st.markdown(full_text)

            # Soft delete / restore
            col_a, col_b = st.columns(2)
            with col_a:
                if not is_del:
                    if st.button(f"🗑️ Delete", key=f"del_{rid}"):
                        try:
                            conn2 = get_conn()
                            soft_delete_report(conn2, rid)
                            conn2.close()
                            st.success("Report marked as deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
            with col_b:
                if is_del:
                    if st.button(f"♻️ Restore", key=f"restore_{rid}"):
                        try:
                            conn2 = get_conn()
                            restore_report(conn2, rid)
                            conn2.close()
                            st.success("Report restored.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Restore failed: {e}")


# ================================================================
# Metrics Tab
# ================================================================
def _render_metrics_tab():
    """📊 Objective Metrics — 评估历史、趋势图、模型分布。"""
    from db import get_conn, get_evals_with_report, get_score_trend, get_total_cost, get_chat_logs, get_chat_rag_summary, get_cost_quality_df

    st.subheader("📊 Objective Metrics History")

    try:
        conn = get_conn()
        evals_df = get_evals_with_report(conn)
        trend_df = get_score_trend(conn)
        total_cost = get_total_cost(conn)
        cost_quality_df = get_cost_quality_df(conn)
        conn.close()
    except Exception as e:
        st.error(f"Could not load metrics: {e}")
        return

    if evals_df.empty:
        st.info("No evaluation data yet. Generate a summary to start collecting metrics.")
        return

    # --- KPI Cards ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Runs", len(evals_df))
    with kpi2:
        avg_ws = evals_df['objective_weighted_score'].dropna().mean()
        st.metric("Avg Weighted Score", f"{avg_ws:.3f}" if not pd.isna(avg_ws) else "N/A")
    with kpi3:
        st.metric("Total Cost", f"${total_cost:.4f}")
    with kpi4:
        latest_model = evals_df.iloc[0]['model_used'] if not evals_df.empty else "N/A"
        st.metric("Latest Model", latest_model)

    st.markdown("---")

    # --- Score Trend Chart ---
    if not trend_df.empty and 'objective_weighted_score' in trend_df.columns:
        trend_valid = trend_df.dropna(subset=['objective_weighted_score']).copy()
        if not trend_valid.empty:
            st.subheader("Weighted Score Trend")
            st.line_chart(
                trend_valid.set_index('created_at')['objective_weighted_score'],
            )

    # --- Model Distribution ---
    if not evals_df.empty:
        st.subheader("Model Distribution")
        model_counts = evals_df['model_used'].value_counts()
        st.bar_chart(model_counts)

    # --- Detailed Table ---
    st.subheader("Evaluation History")
    display_cols = [
        'created_at', 'model_used',
        'factual_consistency', 'completeness', 'coherence',
        'objective_weighted_score', 'report_deleted',
    ]
    available_cols = [c for c in display_cols if c in evals_df.columns]
    st.dataframe(
        evals_df[available_cols],
        hide_index=True,
    )

    # ────────────────────────────────────────────────────────────
    # Phase 2B: Advanced Charts
    # ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Advanced Analytics (Phase 2B)")

    if not cost_quality_df.empty:
        try:
            import plotly.express as px

            col_left, col_right = st.columns(2)

            # ── Chart 1: Cost vs Quality Scatter ──
            with col_left:
                st.markdown("**💰 Cost vs Quality**")
                fig_scatter = px.scatter(
                    cost_quality_df,
                    x="cost_usd",
                    y="objective_weighted_score",
                    color="model_used",
                    hover_data=["created_at", "factual_consistency", "completeness", "coherence"],
                    labels={
                        "cost_usd": "Cost (USD)",
                        "objective_weighted_score": "Weighted Score",
                        "model_used": "Model",
                    },
                    title="Cost vs Quality per Run",
                )
                fig_scatter.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_scatter, use_container_width=True)

            # ── Chart 2: Token Efficiency Trend ──
            with col_right:
                teff = cost_quality_df.dropna(subset=["token_efficiency"])
                if not teff.empty:
                    st.markdown("**⚡ Token Efficiency (Score / 1K Tokens)**")
                    fig_eff = px.line(
                        teff,
                        x="created_at",
                        y="token_efficiency",
                        color="model_used",
                        markers=True,
                        labels={
                            "created_at": "Time",
                            "token_efficiency": "Score / 1K Tokens",
                            "model_used": "Model",
                        },
                        title="Token Efficiency over Time",
                    )
                    fig_eff.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig_eff, use_container_width=True)
                else:
                    st.info("Token efficiency data not yet available.")

            # ── Chart 3: Judge vs SBERT Score Comparison ──
            judge_data = cost_quality_df.dropna(subset=["judge_score"])
            if not judge_data.empty:
                st.markdown("**🧑‍⚖️ Judge-LLM vs SBERT Score Comparison**")
                compare_df = judge_data[["created_at", "model_used",
                                         "objective_weighted_score", "judge_score"]].copy()
                compare_df = compare_df.rename(columns={
                    "objective_weighted_score": "SBERT Weighted",
                    "judge_score": "Judge-LLM",
                })
                compare_melt = compare_df.melt(
                    id_vars=["created_at", "model_used"],
                    value_vars=["SBERT Weighted", "Judge-LLM"],
                    var_name="Metric",
                    value_name="Score",
                )
                fig_cmp = px.line(
                    compare_melt,
                    x="created_at",
                    y="Score",
                    color="Metric",
                    line_dash="Metric",
                    markers=True,
                    labels={"created_at": "Time", "Score": "Score [0-1]"},
                    title="SBERT Weighted Score vs Judge-LLM Score",
                )
                fig_cmp.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Judge results table
                flags_rows = cost_quality_df[
                    cost_quality_df["judge_score"].notna()
                ][["created_at", "model_used", "judge_score"]].copy()
                flags_rows["judge_score"] = flags_rows["judge_score"].apply(
                    lambda v: f"{'✅' if v >= 0.75 else '⚠️' if v >= 0.55 else '❌'} {v:.3f}"
                )
                st.markdown("**Judge-LLM Results**")
                st.dataframe(flags_rows, hide_index=True, use_container_width=True)
            else:
                st.caption("🧑‍⚖️ Judge-LLM scores will appear here after the next summary is generated.")

        except ImportError:
            st.info("Install plotly for advanced charts: `pip install plotly`")
    else:
        st.info("Generate at least one summary to see advanced analytics.")

    # ────────────────────────────────────────────────────────────
    # Chat QA RAG Observability
    # ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Chat QA — RAG Observability")
    st.caption(
        "每轮对话的 RAG 可观测性指标。"
        " Retrieval Confidence（检索置信度）衡量检索到了什么；"
        " Answer Grounding（答案锚定）衡量回答用了多少检索内容。"
    )

    try:
        chat_conn = get_conn()
        chat_df = get_chat_logs(chat_conn)
        ab_rows_raw = get_chat_rag_summary(chat_conn)
        chat_conn.close()
    except Exception as e:
        st.error(f"Could not load chat logs: {e}")
        chat_df = pd.DataFrame()
        ab_rows_raw = []

    if chat_df.empty:
        st.info("No chat QA turns logged yet. Ask a question after generating a summary.")
    else:
        # --- A/B Summary Cards ---
        ab_rows = ab_rows_raw

        if ab_rows:
            rag_on  = next((r for r in ab_rows if r.get("rag_used")), None)
            rag_off = next((r for r in ab_rows if not r.get("rag_used")), None)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Turns w/ RAG",  rag_on["turns"]  if rag_on  else 0)
            with c2:
                st.metric("Turns w/o RAG", rag_off["turns"] if rag_off else 0)
            with c3:
                val = rag_on["avg_grounding"] if rag_on and rag_on.get("avg_grounding") else None
                st.metric("Avg Grounding (RAG ON)",  f"{val:.3f}" if val is not None else "N/A")
            with c4:
                val = rag_on["avg_confidence"] if rag_on and rag_on.get("avg_confidence") else None
                st.metric("Avg Confidence (RAG ON)", f"{val:.3f}" if val is not None else "N/A")

        # --- Grounding Trend ---
        grounding_data = chat_df.dropna(subset=["answer_grounding_score"])
        if not grounding_data.empty:
            st.subheader("Answer Grounding Score over Time")
            st.line_chart(
                grounding_data.set_index("created_at")["answer_grounding_score"]
            )

        # --- Detailed Chat Log Table ---
        st.subheader("Per-Turn Detail")
        chat_display = chat_df[[
            "created_at", "rag_used", "n_chunks_retrieved",
            "retrieval_confidence", "answer_grounding_score",
            "model_used", "question",
        ]].copy()
        # Colour-code grounding for quick visual scanning
        def _grounding_label(v):
            if pd.isna(v):  return "—"
            if v >= 0.65:   return f"✅ {v:.3f}"
            if v >= 0.45:   return f"⚠️ {v:.3f}"
            return              f"❌ {v:.3f}"

        chat_display["answer_grounding_score"] = chat_display["answer_grounding_score"].apply(_grounding_label)
        chat_display["retrieval_confidence"]   = chat_display["retrieval_confidence"].apply(
            lambda v: f"{v:.4f}" if not pd.isna(v) else "—"
        )
        st.dataframe(chat_display, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
