"""
judge_llm.py — LLM-as-a-Judge 异步副轨（Phase 2B）

设计原则（DESIGN.md §11.2）：
  - 主轨不变：SBERT/TF-IDF 指标仍是实时决策唯一信号（低延迟、可确定、低成本）
  - 副轨新增：异步低成本 Judge-LLM（gemini-2.0-flash-lite-001）补充逻辑一致性检测
  - 非阻塞：后台线程执行，执行完毕后回写 DuckDB，不影响用户等待时间

核心指标语义：
  judge_score       [0, 1]  — 综合质量分（0=极差, 1=满分）
  judge_confidence  [0, 1]  — Judge 自身的置信度（对复杂视频会更低）
  judge_flags       JSON    — 风险标志: {hallucination_risk, contradiction_hint,
                                         missing_critical_info, poor_structure}
  judge_reasoning   str     — CoT 推理摘要（调试用）

使用示例：
    from judge_llm import run_judge_async
    run_judge_async(
        report_id="uuid...",
        summary=summary_text,
        transcript=transcript_text,
    )
    # 函数立即返回，Judge 在后台运行，完成后自动回写 DuckDB
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def run_judge_async(
    report_id: str,
    summary: str,
    transcript: str,
    model: str = "gemini/gemini-2.0-flash-lite-001",
) -> None:
    """
    非阻塞入口：立即返回，后台线程执行 Judge-LLM 并回写 DuckDB。

    Args:
        report_id:  DuckDB eval_runs 行的 report_id，用于定位回写目标。
        summary:    生成的摘要文本。
        transcript: 原始 transcript 文本（Judge 的 ground truth）。
        model:      Judge 使用的模型（默认 flash-lite，低成本）。
    """
    t = threading.Thread(
        target=_judge_worker,
        args=(report_id, summary, transcript, model),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a strict AI quality evaluator. Your task is to evaluate the quality of a **generated summary** against an **original transcript**.

Evaluate ONLY based on what is in the transcript. Be critical but fair.

---
## ORIGINAL TRANSCRIPT (ground truth, first 4000 chars)
{transcript_excerpt}

---
## GENERATED SUMMARY (to evaluate)
{summary_excerpt}

---
## EVALUATION CRITERIA

1. **Factual Accuracy** (0-3 pts): Does every factual claim in the summary accurately reflect the transcript? Deduct for hallucinations, wrong names/dates/numbers.
2. **Coverage** (0-3 pts): Does the summary capture all the key points from the transcript? Deduct for major omissions.
3. **Coherence & Structure** (0-2 pts): Is the summary logically organized and easy to follow?
4. **Conciseness** (0-2 pts): Is the summary appropriately concise without padding or repetition?

**Total: 0-10 points**

---
## OUTPUT FORMAT (JSON only, no prose outside the JSON)

```json
{{
  "score": <float 0-10>,
  "confidence": <float 0-1, your confidence in this evaluation>,
  "flags": {{
    "hallucination_risk": <true/false>,
    "contradiction_hint": <true/false>,
    "missing_critical_info": <true/false>,
    "poor_structure": <true/false>
  }},
  "reasoning": "<1-2 sentence summary of key findings>"
}}
```
"""


def _judge_worker(
    report_id: str,
    summary: str,
    transcript: str,
    model: str,
) -> None:
    """后台线程：调用 Judge-LLM，解析结果，回写 DuckDB。"""
    start = time.time()
    judge_score = None
    judge_confidence = None
    judge_flags = "{}"
    judge_reasoning = ""
    judge_latency = 0.0

    try:
        result = _call_judge_llm(summary, transcript, model)
        judge_score      = result.get("score")
        judge_confidence = result.get("confidence")
        judge_flags      = json.dumps(result.get("flags", {}))
        judge_reasoning  = result.get("reasoning", "")

        # 归一化 score: 0-10 → 0-1
        if judge_score is not None:
            judge_score = max(0.0, min(1.0, float(judge_score) / 10.0))
        if judge_confidence is not None:
            judge_confidence = max(0.0, min(1.0, float(judge_confidence)))

    except Exception as e:
        print(f"[Judge] Worker error: {e}")
        judge_reasoning = f"Judge failed: {e}"
    finally:
        judge_latency = round(time.time() - start, 2)

    # 回写 DuckDB
    try:
        from db import get_conn, update_eval_judge
        conn = get_conn()
        update_eval_judge(
            conn=conn,
            report_id=report_id,
            judge_score=judge_score,
            judge_confidence=judge_confidence,
            judge_flags=judge_flags,
            judge_reasoning=judge_reasoning,
            judge_model=model.replace("gemini/", ""),
            judge_latency_sec=judge_latency,
        )
        conn.close()
        score_str = f"{judge_score:.3f}" if judge_score is not None else "N/A"
        print(
            f"[Judge] ✅ Done in {judge_latency}s | "
            f"score={score_str} | report_id={report_id[:8]}..."
        )
    except Exception as e:
        print(f"[Judge] ⚠️ DuckDB write-back failed: {e}")


def _call_judge_llm(
    summary: str,
    transcript: str,
    model: str,
) -> dict:
    """
    调用 Gemini flash-lite 获取结构化评分。

    为控制成本与 latency，截取 transcript 前 4000 字符 + summary 前 2000 字符。
    Judge-LLM 的角色是"交叉验证"而非"完全重评"，截取足够有代表性。
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found for Judge-LLM")

    transcript_excerpt = transcript[:4000] if transcript else ""
    summary_excerpt    = summary[:2000]    if summary    else ""

    prompt = _JUDGE_PROMPT.format(
        transcript_excerpt=transcript_excerpt,
        summary_excerpt=summary_excerpt,
    )

    # 使用 crewai LLM 直接调用（复用已有依赖，无需引入 google-generativeai）
    from crewai import LLM
    llm = LLM(
        model=model,
        api_key=api_key,
        temperature=0.0,   # Judge 需要确定性输出
    )

    raw_response = llm.call(
        messages=[{"role": "user", "content": prompt}]
    )

    return _parse_judge_response(str(raw_response))


def _parse_judge_response(raw: str) -> dict:
    """
    从 LLM 原始文本中提取 JSON。

    容错策略：
      1. 尝试直接 json.loads（LLM 完美遵守格式时）
      2. 用正则提取 ```json ... ``` 代码块
      3. 用正则扫描第一个 { ... } 大括号块
      4. 全部失败时返回默认空结果（不抛异常）
    """
    default = {
        "score": None,
        "confidence": None,
        "flags": {},
        "reasoning": "Parse failed",
    }

    if not raw or not raw.strip():
        return default

    # 尝试1：直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 尝试2：提取 ```json ... ``` 块
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试3：扫描第一个完整 { ... } 块
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 兜底：提取 score 数字
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
    if score_match:
        default["score"] = float(score_match.group(1))
        default["reasoning"] = "Partial parse: score only"
    return default
