import os
import streamlit as st
import requests
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load secrets from Streamlit Cloud ---
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
APP_KEY = st.secrets["APP_KEY"]

# --- Load embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- In-memory vector store ---
chunk_embeddings = []
chunk_texts = []

# --- Chunk utility ---
def chunk_code(code, chunk_size=20):
    lines = code.splitlines()
    return ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

# --- Load and embed all code ---
def load_and_embed_code(folder_path="./codebase"):
    global chunk_embeddings, chunk_texts
    chunk_embeddings.clear()
    chunk_texts.clear()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".py", ".txt")):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code = f.read()
                chunks = chunk_code(code)
                embeddings = embedder.encode(chunks)
                chunk_texts.extend(chunks)
                chunk_embeddings.extend(embeddings)

# --- Retrieve top-k similar chunks ---
def retrieve_code(prompt, top_k=3):
    query_vec = embedder.encode([prompt])
    sim_scores = cosine_similarity(query_vec, chunk_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_k]
    return [chunk_texts[i] for i in top_indices]

# --- Get CIRCUIT Access Token ---
def get_circuit_access_token():
    url = "https://id.cisco.com/oauth2/default/v1/token"
    payload = "grant_type=client_credentials"
    auth = f"{CLIENT_ID}:{CLIENT_SECRET}"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic " + base64_encode(auth)
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json().get("access_token", "")

# --- Base64 encoder ---
def base64_encode(value):
    import base64
    return base64.b64encode(value.encode()).decode()

# --- Generate response using CIRCUIT API ---
def generate_code(user_prompt, retrieved_chunks):
    token = get_circuit_access_token()
    if not token:
        return "❌ Failed to get access token."

    context = "\n\n".join(retrieved_chunks)
    full_prompt = f"""Use the following context:

{context}

Now answer the instruction:
\"\"\"
{user_prompt}
\"\"\""""

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ],
        "user": f'{{"appkey": "{APP_KEY}"}}',
        "stop": ["<|im_end|>"]
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "api-key": token
    }

    url = "https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions"
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Code Assistant (In-Memory)", layout="wide")
st.title("💬 RAG Code Generator (Streamlit Compatible)")
st.markdown("Ask coding questions and get responses based on your local codebase.")

with st.sidebar:
    st.header("📦 Load Codebase")
    if st.button("🔄 Load & Embed Codebase"):
        load_and_embed_code()
        st.success("✅ Codebase loaded and embedded!")

prompt = st.text_area("💡 Enter your instruction:", height=150)

if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    elif not chunk_embeddings:
        st.warning("⚠️ Codebase not loaded. Click the sidebar button.")
    else:
        with st.spinner("⏳ Thinking..."):
            chunks = retrieve_code(prompt)
            output = generate_code(prompt, chunks)
            st.success("✅ Output Generated!")
            st.code(output, language="python")
