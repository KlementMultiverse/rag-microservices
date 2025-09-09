from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ← NEW IMPORT
loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(pages[0].metadata)
print(pages[0].page_content)
for i, page in enumerate(pages):
    print(f"Page {i+1} length: {len(page.page_content)} characters")
# === NEW CODE BELOW ===
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # ← CREATE SPLITTER
chunks = splitter.split_documents(pages)  # ← SPLIT DOCUMENTS
print(f"\nTotal chunks created: {len(chunks)}\n")  # ← PRINT TOTAL
for i, chunk in enumerate(chunks[:3]):  # ← PRINT FIRST 3
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()



# Save chunks to a text file for the embedder
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- CHUNK {i+1} ---\n")
        f.write(chunk.page_content + "\n\n")
