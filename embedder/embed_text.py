import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Load chunks from file
chunks_file = "../document-ingestor/chunks.txt"
texts = []

with open(chunks_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split by chunk header
chunks_raw = content.split("--- CHUNK ")[1:]  # skip empty first part

for chunk in chunks_raw:
    lines = chunk.strip().split("\n")
    if len(lines) > 1:
        # First line is header (e.g., "1 ---"), rest is content
        text = "\n".join(lines[1:]).strip()
        if text:  # only add non-empty
            texts.append(text)

embeddings = model.encode(texts)



print(f"Generated embeddings for {len(texts)} chunks.")

print(f"Embeddings shape: {embeddings.shape}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")



# Save embeddings to file
np.save("embeddings.npy", embeddings)
print("✅ Embeddings saved to embeddings.npy")

# Optional: Save chunk texts too (for debugging)
with open("chunk_texts.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n---\n")
print("✅ Chunk texts saved to chunk_texts.txt")


# Load and verify
loaded_embeddings = np.load("embeddings.npy")
print(f"✅ Loaded embeddings shape: {loaded_embeddings.shape}")
print(f"First 5 values match: {np.allclose(embeddings[0][:5], loaded_embeddings[0][:5])}")
