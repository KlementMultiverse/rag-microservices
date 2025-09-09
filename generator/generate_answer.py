import requests

# vLLM server endpoint
VLLM_API_URL = "http://localhost:5000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def generate_answer(context: str, question: str):
    # Format messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Use ONLY the context below to answer accurately and concisely."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    # Prepare payload
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.0
    }
    
    # Send request
    response = requests.post(VLLM_API_URL, json=payload, headers=HEADERS)
    
    # Handle response
    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer.strip()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Test
sample_context = """
To activate virtual environment, run: source venv/bin/activate on Mac/Linux.
On Windows, use: venv\\Scripts\\activate.
Always activate before installing packages.
"""

sample_question = "How do I activate virtual environment on Mac?"

print(f"❓ Question: {sample_question}\n")
print(f"📚 Context:\n{sample_context}\n")

try:
    answer = generate_answer(sample_context, sample_question)
    print(f"✅ Answer:\n{answer}")
except Exception as e:
    print(f"❌ Error: {e}")
