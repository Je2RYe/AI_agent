"""Quick test script for debugging evaluation performance."""
import time
from evaluation import compute_objective_metrics
from manager import AdaptiveManager

def test_evaluation():
    print("=== Testing Evaluation ===")
    summary = "This is a test summary. It contains important information."
    transcript = "This is a test transcript. It contains important information and more details."
    
    start = time.time()
    print("Computing metrics...")
    metrics = compute_objective_metrics(summary, transcript)
    elapsed = time.time() - start
    
    print(f"\nFactual Consistency: {metrics['factual_consistency']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"Coherence: {metrics['coherence']:.4f}")
    print(f"Time taken: {elapsed:.2f}s")

def test_manager():
    print("\n=== Testing Manager ===")
    manager = AdaptiveManager()
    print(f"Run count: {manager.get_run_count()}")
    
    ws = manager.calculate_weighted_score(0.85, 0.72, 0.91)
    print(f"Weighted score: {ws}")
    
    cost = manager.calculate_cost('gemini-2.0-flash-lite-001', 10000, 5000)
    print(f"Cost: ${cost}")

if __name__ == "__main__":
    test_manager()
    test_evaluation()
    print("\n✓ All tests passed")
