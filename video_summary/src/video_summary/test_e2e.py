#!/usr/bin/env python
"""End-to-end pipeline test: YouTube URL → transcribe → summarize → evaluate → log"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from crew import VideoSummary
from manager import AdaptiveManager
from evaluation import compute_objective_metrics

TEST_URL = "https://www.youtube.com/watch?v=wjr4iFzlLjo"

def main():
    manager = AdaptiveManager()
    print("=== E2E Test Start ===")
    selected_model = manager.decide_model()
    remaining, used, limit = manager.get_resource_status()
    print(f"Model selected: {selected_model}")
    print(f"Resource status: remaining={remaining:.2%}, used=${used:.4f}, limit=${limit:.2f}")

    inputs = {"content": TEST_URL}
    print(f"\n--- Running with model: {selected_model} ---")
    start_time = time.time()

    video_summary_crew = VideoSummary(model_name=selected_model)
    crew_instance = video_summary_crew.create_summarization_crew()
    result = crew_instance.kickoff(inputs=inputs)
    latency = time.time() - start_time

    generated_text = str(result)
    print(f"Latency: {latency:.2f}s")
    print(f"Result length: {len(generated_text)} chars")

    # Token usage
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if hasattr(result, "token_usage"):
        usage = result.token_usage
        print(f"Token usage raw: {usage}")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
    else:
        total_tokens = len(generated_text) // 4 + 1000
        prompt_tokens = 1000
        completion_tokens = total_tokens - 1000
    print(f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

    cost = manager.calculate_cost(selected_model, prompt_tokens, completion_tokens)
    print(f"Cost: ${cost:.6f}")

    # Get transcript from first task output
    transcript_text = ""
    try:
        transcript_text = crew_instance.tasks[0].output.raw
        print(f"Transcript length: {len(transcript_text)} chars")
    except Exception as e:
        print(f"Could not get transcript: {e}")

    # Save files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = "history_reports"
    os.makedirs(history_dir, exist_ok=True)

    clean_model = selected_model.replace("/", "-").replace(":", "")
    report_path = os.path.join(history_dir, f"summary_{timestamp}_{clean_model}.txt")
    transcript_path = os.path.join(history_dir, f"transcript_{timestamp}_{clean_model}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generated_text)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print(f"Report saved: {report_path}")
    print(f"Transcript saved: {transcript_path}")

    # Objective metrics
    print("\n--- Computing Objective Metrics ---")
    if generated_text and transcript_text:
        metrics = compute_objective_metrics(generated_text, transcript_text)
        fc = metrics["factual_consistency"]
        cp = metrics["completeness"]
        ch = metrics["coherence"]
        print(f"Factual Consistency: {fc:.4f}")
        print(f"Completeness:        {cp:.4f}")
        print(f"Coherence:           {ch:.4f}")

        ws = manager.calculate_weighted_score(fc, cp, ch)
        print(f"Weighted Score:      {ws:.4f}")
    else:
        fc = cp = ch = None
        print("SKIP: missing summary or transcript text")

    # Log execution
    manager.log_execution(
        model_used=selected_model,
        latency=latency,
        total_tokens=total_tokens,
        cost=cost,
        report_path=report_path,
        transcript_path=transcript_path,
        factual_consistency=fc,
        completeness=cp,
        coherence=ch,
    )
    print("\n--- Logged to system_metrics.csv ---")

    # Read back CSV
    import pandas as pd
    csv_path = "system_metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"CSV rows: {len(df)}")
        print(f"CSV columns ({len(df.columns)}): {list(df.columns)}")
        print("\nLast row:")
        print(df.tail(1).to_string())
    else:
        print("WARNING: system_metrics.csv not found!")

    # Print summary snippet
    print(f"\n--- Summary Preview (first 500 chars) ---")
    print(generated_text[:500])

    print("\n=== E2E Test Complete ===")


if __name__ == "__main__":
    main()
