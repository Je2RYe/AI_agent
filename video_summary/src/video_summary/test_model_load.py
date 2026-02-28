"""Test if SentenceTransformer model loads correctly."""
import time
from sentence_transformers import SentenceTransformer

print("Step 1: Loading model...")
start = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
load_time = time.time() - start
print(f"✓ Model loaded in {load_time:.2f}s")

print("\nStep 2: Testing encoding...")
start = time.time()
test_text = ["This is a test sentence.", "This is another sentence."]
embeddings = model.encode(test_text)
encode_time = time.time() - start
print(f"✓ Encoded {len(test_text)} sentences in {encode_time:.2f}s")
print(f"  Embedding shape: {embeddings.shape}")

print("\nStep 3: Testing with more sentences...")
start = time.time()
many_sentences = ["Sentence number " + str(i) for i in range(20)]
embeddings = model.encode(many_sentences)
encode_time = time.time() - start
print(f"✓ Encoded {len(many_sentences)} sentences in {encode_time:.2f}s")

print("\n✓ All model tests passed!")
