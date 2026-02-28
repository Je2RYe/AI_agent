"""验证新 completeness 算法效果（只测 completeness，避免长视频 FC encoding OOM）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "video_summary", "src", "video_summary"))

from evaluation import _split_sentences, ObjectiveEvaluator

pairs = [
    ('有RAG短视频 #2', r'history_reports\summary_20260228_201147_gemini-2.5-flash.txt', r'history_reports\transcript_20260228_201147_gemini-2.5-flash.txt'),
    ('无RAG短视频 #3', r'history_reports\summary_20260228_201313_gemini-2.5-flash.txt', r'history_reports\transcript_20260228_201313_gemini-2.5-flash.txt'),
    ('无RAG长视频 #4', r'history_reports\summary_20260228_201952_gemini-2.5-flash.txt', r'history_reports\transcript_20260228_201952_gemini-2.5-flash.txt'),
    ('有RAG长视频 #5', r'history_reports\summary_20260228_202214_gemini-2.5-flash.txt', r'history_reports\transcript_20260228_202214_gemini-2.5-flash.txt'),
]

OLD_RESULTS = {
    '有RAG短视频 #2': 0.4000,
    '无RAG短视频 #3': 0.3714,
    '无RAG长视频 #4': 0.1429,
    '有RAG长视频 #5': 0.1209,
}

print("=" * 70)
print("Completeness 算法 A/B 对比 (旧 vs 新)")
print("=" * 70)

evaluator = ObjectiveEvaluator()

for label, sp, tp in pairs:
    summary = open(sp, encoding='utf-8').read()
    transcript = open(tp, encoding='utf-8').read()
    t_sents = _split_sentences(transcript)
    s_sents = _split_sentences(summary)
    
    top_k = evaluator._get_dynamic_top_k(len(t_sents))
    new_comp = evaluator.completeness(summary, transcript)
    old_comp = OLD_RESULTS[label]
    delta = new_comp - old_comp
    pct = (delta / old_comp * 100) if old_comp else 0
    
    print(f'\n--- {label} ---')
    print(f'  Transcript: {len(t_sents):>4d} sentences | Summary: {len(s_sents):>3d} sentences')
    print(f'  Dynamic Top-K: {top_k:>3d} (from {len(t_sents)} total)')
    print(f'  OLD Comp: {old_comp:.4f}  →  NEW Comp: {new_comp:.4f}  (Δ {delta:+.4f}, {pct:+.1f}%)')

print("\n" + "=" * 70)
