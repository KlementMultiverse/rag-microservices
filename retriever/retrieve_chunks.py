
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("../vector-db/faiss_index.bin")
print(f"✅ Loaded index with {index.ntotal} vectors")

# Load chunk texts
with open("../embedder/chunk_texts.txt", "r", encoding="utf-8") as f:
    chunk_texts = f.read().strip().split("---\n")
    chunk_texts = [text.strip() for text in chunk_texts if text.strip()]
print(f"✅ Loaded {len(chunk_texts)} chunk texts")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def retrieve_similar_chunks(query: str, k: int = 3):
    # Embed query
    query_embedding = model.encode([query])
    
    # Search FAISS
    D, I = index.search(query_embedding.astype('float32'), k)
    
    # Format results
    results = []
    for i in range(len(I[0])):
        idx = I[0][i]
        distance = D[0][i]
        text = chunk_texts[idx] if idx < len(chunk_texts) else "N/A"
        results.append({
            "rank": i+1,
            "chunk_id": idx,
            "distance": float(distance),
            "text": text[:200] + "..." if len(text) > 200 else text
        })
    
    return results

# Test
test_query = "How do I activate virtual environment?"
print(f"\n🔍 Searching for: '{test_query}'\n")

results = retrieve_similar_chunks(test_query, k=3)
for result in results:
    print(f"--- RANK {result['rank']} (Distance: {result['distance']:.4f}) ---")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text']}\n")
