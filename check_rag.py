import duckdb
import pandas as pd

conn = duckdb.connect('video_summary.duckdb')
result = conn.execute('''
    SELECT r.created_at, r.model_used, r.latency_sec, r.total_tokens, r.cost_usd,
           e.factual_consistency, e.completeness, e.coherence, e.objective_weighted_score,
           e.rag_used, e.rag_total_chunks, e.rag_used_chunks, e.rag_coverage_pct
    FROM reports r 
    LEFT JOIN eval_runs e ON r.report_id = e.report_id 
    WHERE r.is_deleted = FALSE 
    ORDER BY r.created_at DESC 
    LIMIT 5
''').fetchall()

df = pd.DataFrame(result, columns=[
    '时间', '模型', '延迟', 'tokens', 'cost', 
    'FC', 'Comp', 'Coh', 'Score', 'RAG', '总chunks', '用chunks', '覆盖率'
])

print("=" * 100)
print("最近5次测试对比")
print("=" * 100)
print(df.to_string(index=False))

print("\n" + "=" * 100)
print("详细RAG分析")
print("=" * 100)
for i, row in enumerate(reversed(list(df.itertuples())), 1):
    time_str = row.时间.strftime("%H:%M:%S")
    rag_str = "✅启用" if row.RAG else "❌禁用"
    chunks_str = f"{row.用chunks}/{row.总chunks}" if row.RAG else "N/A"
    cov_str = f"{row.覆盖率:.1%}" if row.RAG and row.覆盖率 else "N/A"
    print(f"#{i} [{time_str}] {rag_str} | Score={row.Score:.4f} | FC={row.FC:.3f} | "
          f"Comp={row.Comp:.3f} | tokens={row.tokens} | chunks={chunks_str} | 覆盖={cov_str}")

conn.close()
