from fastapi import FastAPI
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retriever.retrieve_chunks import retrieve_similar_chunks
from generator.generate_answer import generate_answer

app = FastAPI(title="🧠 RAG API Gateway")

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "RAG Gateway is running"}

@app.post("/query")
def query_rag(question: str):
    # Retrieve top 3 chunks
    retrieved_chunks = retrieve_similar_chunks(question, k=3)
    
    # Combine into context
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    # Generate answer
    answer = generate_answer(context, question)
    
    # Return structured response
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [
            {
                "rank": chunk["rank"],
                "chunk_id": chunk["chunk_id"],
                "distance": chunk["distance"],
                "text": chunk["text"]
            }
            for chunk in retrieved_chunks
        ]
    }
