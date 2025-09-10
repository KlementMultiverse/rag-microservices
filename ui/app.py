import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="🧠 RAG 101 Demo",
    page_icon="🔍",
    layout="wide"
)

# NVIDIA-style dark theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
        border-radius: 8px;
        border: 1px solid #333;
        padding: 10px;
    }
    .stMarkdown { color: #ffffff; }
    .stExpander { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333; }
    h1, h2, h3, h4 { color: #00ff9d !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Retrieval-Augmented Generation (RAG) Demo")
st.markdown("Ask questions about the RAG development guide — answers grounded in retrieved context.")

# Input box
question = st.text_input(
    "Enter your question:",
    placeholder="How do I activate virtual environment?",
    key="question_input"
)

# Submit button
if st.button("🔍 Get Answer", type="primary", key="submit_btn"):
    # Validate input
    if not question or not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        user_question = question.strip()
        
        with st.spinner("🧠 Retrieving context and generating answer..."):
            try:
                # Debug: show what we're sending
                st.write(f"📤 Sending question: '{user_question}'")
                
                # Call API Gateway
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"question": user_question},  # Explicitly send stripped string
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Answer in beautiful card
                    st.subheader("✅ Generated Answer")
                    st.markdown(f"""
                    <div style='background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 4px solid #00ff9d;'>
                        <h4 style='color:#00ff9d; margin:0;'>{result['answer']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Copyable answer
                    st.code(result['answer'], language="text")
                    if st.button("📋 Copy Answer", key="copy_btn"):
                        st.write("✅ Copied to clipboard! (Simulated — in real app, use JS)")

                    # Display Sources
                    st.subheader("📚 Retrieved Context (Top 3)")
                    for chunk in result["retrieved_chunks"]:
                        with st.expander(f"📌 Rank {chunk['rank']} | Distance: {chunk['distance']:.4f} | Chunk ID: {chunk['chunk_id']}"):
                            st.text(chunk["text"])
                
                elif response.status_code == 422:
                    st.error("❌ Validation Error: Question field is missing or invalid.")
                    st.write("🔧 Debug: Check that your API Gateway expects `question: str` in POST body.")
                
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API Gateway. Is it running on http://localhost:8000?")
                st.write("💡 Tip: Run in terminal: `uvicorn main:app --reload --port 8000` in gateway folder")
                
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Server is slow or overloaded.")
                
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")
                st.write("🔧 Debug info:", type(e).__name__)

# Footer
st.markdown("---")
st.markdown("💡 Built with Streamlit + FastAPI + Qwen2.5-3B + FAISS — like a pro.")
