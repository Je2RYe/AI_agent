#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
import streamlit as st
from crewai import Agent, Crew, Process, Task
import os
import time
import threading
import pandas as pd
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crew import VideoSummary
from manager import AdaptiveManager

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv()

if 'manager' not in st.session_state:
    st.session_state.manager = AdaptiveManager()

manager = st.session_state.manager

def run_summarization(inputs, done_flag, model_name):
    """
    运行 Crew 的工作线程函数
    """
    video_summary_crew = VideoSummary(model_name=model_name)
    start_time = time.time()
    
    try:
        crew_instance = video_summary_crew.create_summarization_crew()
        # 1. 生成报告
        result = video_summary_crew.create_summarization_crew().kickoff(inputs=inputs)
        generated_text = str(result)
        done_flag['success'] = True
        
        transcription_text = ""
        try:
            # 尝试获取第一个任务(Transcriber)的输出
            transcription_text = crew_instance.tasks[0].output.raw
        except Exception as e:
            print(f"Warning: Could not capture transcription: {e}")
            transcription_text = "Transcription not available."

        # 3. 准备文件名和路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = "history_reports"
        os.makedirs(history_dir, exist_ok=True)
        
        with open("Video_Summary.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)

        # 2. 历史存档 (Persistence)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = "history_reports"
        os.makedirs(history_dir, exist_ok=True)
        
        clean_model_name = model_name.replace("/", "-").replace(":", "")
        # 保存路径
        transcript_filename = f"transcript_{timestamp}_{clean_model_name}.txt" # [新增]
        saved_filename = f"summary_{timestamp}_{clean_model_name}.txt"
        saved_path = os.path.join(history_dir, saved_filename)
        transcript_path = os.path.join(history_dir, transcript_filename) # [新增]

        with open(saved_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)

        done_flag['report_path'] = saved_path
        done_flag['transcript_path'] = transcript_path

        # 3. 提取 Token 和 Cost
        if hasattr(result, 'token_usage'):
            done_flag['token_usage'] = result.token_usage
        else:
            # Fallback
            est_output = len(generated_text) / 4
            est_input = 1000
            done_flag['token_usage'] = {'total_tokens': est_input+est_output, 'prompt_tokens': est_input, 'completion_tokens': est_output}
            
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
    done_flag = {'finished': False, 'latency': 0, 'success': True, 'error': None}
    
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
        prompt_tokens = 0 
        completion_tokens = 0
        total_tokens = 0
        
        # 兼容性处理
        try:
            prompt_tokens = raw_usage.get('prompt_tokens', 0)
            completion_tokens = raw_usage.get('completion_tokens', 0)
            total_tokens = raw_usage.get('total_tokens', 0)
        except:
            prompt_tokens = getattr(raw_usage, 'prompt_tokens', 0)
            completion_tokens = getattr(raw_usage, 'completion_tokens', 0)
            total_tokens = getattr(raw_usage, 'total_tokens', 0)

        cost = manager.calculate_cost(selected_model, prompt_tokens, completion_tokens)
        st.sidebar.success(f"💰 Cost: ${cost:.5f}")

        # === 判断是否是第 3 次 ===
        current_run_count = manager.get_run_count() + 1 
        is_rating_cycle = (current_run_count % 2 == 0)
        
        is_rating_cycle = True
        current_run_count = manager.get_run_count() + 1

        st.session_state['last_run_data'] = {
            'model': selected_model,
            'latency': done_flag['latency'],
            'tokens': total_tokens,
            'cost': cost,
            'report_path': done_flag['report_path'],
            'transcript_path': done_flag.get('transcript_path'),
            'is_rating_cycle': is_rating_cycle
        }
        
        if is_rating_cycle:
            # [关键修复 3] 如果是评分周期，开启等待状态，并强制 Rerun 刷新 UI 以显示评分框
            st.session_state['waiting_for_rating'] = True
            st.rerun() 
        else:
            # 非评分周期，自动记录
            manager.log_execution(
                model_used=selected_model,
                latency=done_flag['latency'],
                user_rating=0, # 默认值
                ai_rating=0, 
                overall_score=0, 
                tokens=total_tokens,
                cost=cost,
                report_path=done_flag['report_path']
             )
            st.session_state['waiting_for_rating'] = False
            st.info(f"ℹ️ Run #{current_run_count} auto-logged. (Detailed review in {3 - (current_run_count % 2)} runs)")

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
            from manager import MODEL_TIERS 
            selected_model_override = st.selectbox("Force Model:", MODEL_TIERS)
        
        # 2. (可选) 显示当前系统状态
        if st.button("View Metrics CSV"):
            if os.path.exists("system_metrics.csv"):
                try:
                    df = pd.read_csv("system_metrics.csv")
                    st.dataframe(df.tail(5)) # 显示最近5条
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
            else:
                st.write("No history yet.")

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


    st.title("📄 " + t["summary_title"])
    summary_file_path = "Video_Summary.txt"
    
    if os.path.exists("Video_Summary.txt"):
        with open("Video_Summary.txt", "r", encoding="utf-8") as f:
            summary_content = f.read()
        st.subheader(t["summary_title"])
        st.markdown(summary_content)
        
        st.markdown("---")
        
        # === [Step C: Feedback Loop] ===
        run_data = st.session_state.get('last_run_data', {})
        is_cycle = run_data.get('is_rating_cycle', False)
        # 必须同时满足：是评分周期 且 处于等待评分状态
        should_show_rating = st.session_state.get('waiting_for_rating', False) and is_cycle
        
        if should_show_rating:
            st.subheader("📊How would you rate the quality of this AI-generated report? ")
            
            with st.form("rating_form"):
                #st.markdown(f"**Run Details:** Model `{run_data.get('model')}` | Cost `${run_data.get('cost'):.4f}`")
                user_rating = st.slider("User Satisfaction (1-10)", 1, 10, 7)
                submitted = st.form_submit_button("Submit Feedback")
                
                if submitted:
                    ai_score = -1 
                    
                    # === [触发条件] 如果用户给分 <= 5，启动 AI 严格审查 ===
                    with st.spinner("⚠️ Low score detected. AI Auditor is reading local files..."):
                            
                        # 1. 读取本地保存的 Summary
                        summary_content = ""
                        if run_data.get('report_path') and os.path.exists(run_data['report_path']):
                            with open(run_data['report_path'], 'r', encoding='utf-8') as f:
                                summary_content = f.read()
                        else:
                            summary_content = "Error: Summary file not found."

                        # 2. 读取本地保存的 Transcript [关键]
                        transcript_content = ""
                        if run_data.get('transcript_path') and os.path.exists(run_data['transcript_path']):
                            with open(run_data['transcript_path'], 'r', encoding='utf-8') as f:
                                transcript_content = f.read()
                        else:
                            transcript_content = "Error: Transcription file not found. Cannot verify facts."

                        # 3. 启动 Evaluator Crew 并注入双份数据
                        audit_crew_instance = VideoSummary() 
                        
                        # [这里实现了你的要求] 将转录和总结都传进去
                        eval_inputs = {
                            'transcription': transcript_content, # 对应 prompt 中的 {transcription}
                            'summary': summary_content           # 对应 prompt 中的 {summary}
                        }
                        
                        try:
                            # 运行审查任务
                            eval_result = audit_crew_instance.create_evaluation_crew().kickoff(inputs=eval_inputs)
                            
                            # 解析 JSON 输出 (假设 Agent 严格遵循了 JSON 格式)
                            # 注意：CrewAI 返回的是 String，有时候包含 ```json ... ```，需要清洗
                            eval_text = str(eval_result).strip()
                            # 简单的清洗逻辑，提取 JSON 部分
                            if "```json" in eval_text:
                                eval_text = eval_text.split("```json")[1].split("```")[0].strip()
                            elif "```" in eval_text:
                                eval_text = eval_text.split("```")[1].split("```")[0].strip()
                            
                            import json
                            eval_json = json.loads(eval_text)
                            
                            ai_score = eval_json.get("final_score", 5)
                            critique = eval_json.get("audit_critique", "No details provided.")
                            
                            # 显示详细的 AI 审计报告
                            st.error(f"🤖 AI Auditor Score: {ai_score}/10")
                            with st.expander("View Auditor's Critique (Why did I lose points?)"):
                                st.markdown(critique)
                                
                        except Exception as e:
                            st.warning(f"AI Audit failed to parse: {e}")
                            ai_score = 5 # Fallback
                    
                    # === 计算 Overall Score ===
                    if ai_score != -1:
                        overall_score = (0.5 * ai_score) + (0.5 * user_rating)
                    else:
                        overall_score = user_rating
                    
                    # 记录进 CSV
                    manager.log_execution(
                        model_used=run_data.get('model'),
                        latency=run_data.get('latency'),
                        user_rating=user_rating,
                        ai_rating=ai_score,
                        overall_score=overall_score,
                        tokens=run_data.get('tokens'),
                        cost=run_data.get('cost'),
                        report_path=run_data.get('report_path')
                    )
                    
                    st.success(f"✅ Cycle Review Complete! Overall Score: {overall_score:.1f}")
                    st.session_state['waiting_for_rating'] = False
                    st.rerun()

        st.markdown("---")
        st.subheader(t["chat_title"])

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(t["chat_input"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    video_summary_crew = VideoSummary()
                    inputs = {
                        'summary': summary_content,
                        'user_message': prompt
                    }
                    chat_crew = video_summary_crew.create_chat_crew()
                    response = chat_crew.kickoff(inputs=inputs)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info(t["summary_info"])


if __name__ == "__main__":
    main()
