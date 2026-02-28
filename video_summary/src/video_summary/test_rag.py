"""Quick smoke test for rag_engine.py"""
from rag_engine import (
    RecursiveChunker, HybridRetriever, ContextOrchestrator,
    should_use_rag, run_rag_pipeline, build_enhanced_prompt,
)

# --- Test 1: RecursiveChunker ---
chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
text = "This is sentence one about machine learning. This is sentence two about deep learning. " * 20
chunks = chunker.chunk_transcript(text)
print(f"Test 1 PASS: {len(chunks)} chunks from {len(text)} chars")

# --- Test 2: should_use_rag ---
short_text = "Hello world. " * 10
long_text = "Hello world. " * 3000
assert not should_use_rag(short_text), "Short text should NOT trigger RAG"
assert should_use_rag(long_text), "Long text should trigger RAG"
print("Test 2 PASS: should_use_rag threshold works")

# --- Test 3: Full pipeline (with model loading) ---
test_transcript = (
    "Artificial intelligence is transforming industries. " * 50
    + "Deep learning models require large amounts of data. " * 50
    + "Natural language processing enables text understanding. " * 50
)
result = run_rag_pipeline(test_transcript)
ru = result["rag_used"]
nu = result["n_chunks_used"]
nt = result["n_chunks_total"]
cov = result["rag_coverage_pct"]
print(f"Test 3 PASS: rag_used={ru}, chunks={nu}/{nt}, coverage={cov:.0%}")
print(f"  Head pinned: {result['pinned_head']}, Tail pinned: {result['pinned_tail']}")

# --- Test 4: build_enhanced_prompt ---
prompt = build_enhanced_prompt(result)
assert len(prompt) > 100
print(f"Test 4 PASS: Enhanced prompt length = {len(prompt)} chars")

# --- Test 5: Non-RAG path ---
short_result = run_rag_pipeline("Short video about cats.")
assert not short_result["rag_used"]
print(f"Test 5 PASS: Short text bypasses RAG (rag_used={short_result['rag_used']})")

print()
print("=== ALL RAG ENGINE TESTS PASSED ===")
