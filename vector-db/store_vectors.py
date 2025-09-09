import faiss
import numpy as np
# Load embeddings
embeddings = np.load("../embedder/embeddings.npy")

# Get dimension
d = embeddings.shape[1]  # 384

#Creates a flat (brute-force) index that uses L2 (Euclidean) distance for similarity search.
index = faiss.IndexFlatL2(d)

index.add(embeddings.astype('float32'))

# Save index
faiss.write_index(index, "faiss_index.bin")
print(f"✅ Saved index with {index.ntotal} vectors")

# Test: search for first vector (should return itself)
query = embeddings[0].astype('float32').reshape(1, -1)
D, I = index.search(query, k=3)

print("Distances:", D)
print("Indices:", I)
