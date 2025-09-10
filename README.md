🧠 RAG Microservices — Production-Ready RAG System

A complete, production-style Retrieval-Augmented Generation (RAG) system built with microservice architecture. Features local LLM inference via vLLM and Qwen2.5-3B, with easy swapping for any LLM API.
✨ Features

    Microservice Architecture: Each component is isolated and independently scalable
    Local LLM Inference: Uses Qwen2.5-3B via vLLM (easily replaceable)
    Vector Search: FAISS-powered semantic search
    Modern UI: NVIDIA-inspired Streamlit interface
    API Gateway: FastAPI orchestration layer
    Production Ready: Fully containerizable and cloud-deployable

🏗️ System Architecture

PDF → Load → Split → Embed → Store → Retrieve → Generate → Answer

The system processes documents through a complete pipeline where each step is handled by a dedicated microservice.
📂 Project Structure

rag-microservices/
├── document-ingestor/      # Load PDF → split into chunks
├── embedder/              # Generate embeddings from chunks
├── vector-db/             # Store embeddings in FAISS index
├── retriever/             # Search FAISS for relevant chunks
├── generator/             # Generate answer using Qwen via vLLM
├── gateway/               # API Gateway — glues everything together
├── ui/                    # Streamlit UI — NVIDIA-style frontend
└── requirements.txt       # All Python dependencies

🧩 Component Overview
1️⃣ Document Ingestor (document-ingestor/ingest_pdf.py)

    Purpose: Loads PDF files and splits them into manageable chunks
    Input: sample.pdf
    Output: chunks.txt
    Key Libraries: PyPDFLoader, RecursiveCharacterTextSplitter

2️⃣ Embedder (embedder/embed_text.py)

    Purpose: Converts text chunks into vector embeddings
    Model: sentence-transformers/all-MiniLM-L6-v2
    Output: embeddings.npy (384-dimensional vectors)
    Key Libraries: sentence-transformers, numpy

3️⃣ Vector Database (vector-db/store_vectors.py)

    Purpose: Builds and stores FAISS search index
    Index Type: IndexFlatL2 (Euclidean distance)
    Output: faiss_index.bin
    Key Libraries: faiss-cpu

4️⃣ Retriever (retriever/retrieve_chunks.py)

    Purpose: Semantic search for relevant document chunks
    Input: User question
    Output: Ranked chunks with relevance scores
    Key Libraries: sentence-transformers, faiss, numpy

5️⃣ Generator (generator/generate_answer.py)

    Purpose: Generates answers using context and question
    LLM: Qwen/Qwen2.5-3B-Instruct via vLLM
    Protocol: OpenAI-compatible API
    Key Libraries: requests

6️⃣ Gateway (gateway/main.py)

    Purpose: FastAPI orchestrator connecting all services
    Endpoints:
        POST /query - Main query endpoint
        GET /health - Health check
    Key Libraries: fastapi, uvicorn

7️⃣ UI (ui/app.py)

    Purpose: User-friendly Streamlit interface
    Style: NVIDIA-inspired dark theme
    Features: Question input, answer display, source citations
    Key Libraries: streamlit, requests

🚀 Quick Start
Prerequisites

    Docker (for vLLM)
    Python 3.8+
    CUDA-compatible GPU (recommended)

Step 1: Start vLLM Server

docker run -it --rm --gpus all \
  -p 5000:5000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.6.6 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 5000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --enforce-eager \
  --max-num-seqs 1

Wait for: Uvicorn running on http://0.0.0.0:5000
Step 2: Setup Python Environment

cd ~/rag-microservices
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Step 3: Start API Gateway

cd gateway
uvicorn main:app --reload --port 8000

Wait for: Application startup complete
Step 4: Start UI

cd ui
streamlit run app.py

Open your browser to: http://localhost:8501
🔄 LLM Integration Options

The system supports multiple LLM providers. Edit generator/generate_answer.py:
OpenAI Integration

def generate_answer(context: str, question: str):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": "Bearer YOUR_OPENAI_API_KEY",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Use ONLY the context below to answer accurately."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            "max_tokens": 150,
            "temperature": 0.0
        }
    )

Mistral API Integration

headers = {
    "Authorization": "Bearer YOUR_MISTRAL_API_KEY",
    "Content-Type": "application/json"
}
url = "https://api.mistral.ai/v1/chat/completions"
# Use model: "mistral-small-latest"

Ollama Integration (Local)

url = "http://localhost:11434/api/chat"
json={
    "model": "mistral",
    "messages": messages,
    "stream": False
}

🧪 Testing the System
Sample Questions (Should Work)

    "What command creates the ingest_pdf.py file?"
    "What is the name of the text splitter used?"
    "What package do I need to install to use PyPDFLoader?"
    "How do I activate virtual environment on Mac?"
    "What decorator is used to create a POST endpoint in FastAPI?"

Questions That Should Return "I don't know"

    "What is the capital of France?"
    "Explain quantum physics."
    "Who is the president of Mars?"

The system should refuse to hallucinate for questions outside the document context.
🛠️ Troubleshooting
Issue 	Solution
CUDA error in vLLM 	Reboot system and restart vLLM container
422 Unprocessable Entity 	Check FastAPI endpoint uses question: str = Body(..., embed=True)
500 Internal Server Error 	Convert numpy.int64 to int(idx) in retriever
Connection refused in UI 	Ensure gateway is running on port 8000
📁 File Structure Details

rag-microservices/
├── document-ingestor/
│   └── ingest_pdf.py           # PDF processing and chunking
├── embedder/
│   └── embed_text.py           # Text embedding generation
├── vector-db/
│   └── store_vectors.py        # FAISS index creation
├── retriever/
│   └── retrieve_chunks.py      # Semantic search
├── generator/
│   └── generate_answer.py      # LLM answer generation
├── gateway/
│   └── main.py                 # FastAPI orchestration
├── ui/
│   └── app.py                  # Streamlit interface
└── requirements.txt            # Python dependencies

🐳 Docker Support

Full containerization support is available. Each microservice can be containerized independently for scalable deployment.
📝 Requirements

Install all dependencies:

pip install -r requirements.txt

Key packages include:

    fastapi
    uvicorn
    streamlit
    sentence-transformers
    faiss-cpu
    numpy
    requests

🤝 Contributing

    Fork the repository
    Create your feature branch
    Commit your changes
    Push to the branch
    Create a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Built with vLLM for efficient LLM inference
    Uses Qwen2.5-3B for local language generation
    Powered by FAISS for vector similarity search
    UI inspired by NVIDIA's design language

