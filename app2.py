import os
import chromadb
from uuid import uuid4
import streamlit as st
import base64
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import streamlit as st



# ----------------------------
# 🔐 Credentials and Constants
# ----------------------------

CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
APP_KEY = st.secrets["APP_KEY"]
CIRCUIT_MODEL_NAME = "gpt-4.1"
API_VERSION = "2025-01-01-preview"

# ----------------------------
# 🔐 Get Cisco CIRCUIT Token
# ----------------------------
def get_circuit_access_token(client_id, client_secret):
    url = "https://id.cisco.com/oauth2/default/v1/token"
    payload = "grant_type=client_credentials"
    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded}"
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.json().get("access_token", None)

# ----------------------------
# 🔁 Chunking Utility
# ----------------------------
def chunk_code(code, chunk_size=20):
    lines = code.splitlines()
    return ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

# ----------------------------
# 🧠 Load and Embed Codebase
# ----------------------------
def load_and_embed_code(folder_path="./codebase"):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".py", ".txt")):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code = f.read()
                chunks = chunk_code(code)
                embeddings = embedder.encode(chunks).tolist()
                ids = [str(uuid4()) for _ in chunks]
                collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# ----------------------------
# 🔍 Retrieve Relevant Chunks
# ----------------------------
def retrieve_code(prompt, top_k=3):
    query_vec = embedder.encode([prompt])[0].tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    return list(set(results["documents"][0]))

# ----------------------------
# 🧠 Generate Code via CIRCUIT API
# ----------------------------
def generate_code(user_prompt, retrieved_chunks):
    access_token = get_circuit_access_token(CLIENT_ID, CLIENT_SECRET)

    if not access_token:
        return "❌ Failed to get access token."

    context = "\n\n".join(retrieved_chunks)
    full_prompt = f"""
Use the following context from the project:

{context}

Based on the above, perform this instruction:
\"\"\"
{user_prompt}
\"\"\"
""".strip()

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
        "api-key": access_token
    }

    api_url = f"https://chat-ai.cisco.com/openai/deployments/{CIRCUIT_MODEL_NAME}/chat/completions"

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------------------
# 💾 Initialize Chroma and Embedder
# ----------------------------
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(name="codebase")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 🌐 Streamlit UI
# ----------------------------
st.set_page_config(page_title="Dynamic Customer KPIs Traffic Model Generation using RAG", layout="wide")
st.title("Dynamic Customer KPIs Traffic Model Generation using RAG")
st.markdown("Give a natural prompt and get code based on your project context!")

with st.sidebar:
    st.header("📦 Codebase Loader")
    if st.button("🔄 Load & Embed Codebase"):
        load_and_embed_code()
        st.success("✅ Codebase embedded into Chroma DB.")

prompt = st.text_area("💡 Enter your instruction:", height=150)

if st.button("🚀 Generate Output"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        st.info("⏳ Thinking...")
        relevant_chunks = retrieve_code(prompt)
        if not relevant_chunks:
            st.warning("⚠️ No relevant context found.")
        else:
            output = generate_code(prompt, relevant_chunks)
            st.success("✅ Output Generated!")
            st.code(output, language="python")
