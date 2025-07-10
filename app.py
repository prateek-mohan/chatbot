import os
import torch
import chromadb
from uuid import uuid4
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(name="codebase")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load TinyLlama codegen model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="float32"
)

codegen_model = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.3
)

# Function to split code or text into chunks
def chunk_code(code, chunk_size=20):
    lines = code.splitlines()
    return ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

# Load and embed codebase (supports .py and .txt)
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

# Retrieve top-k most relevant chunks
def retrieve_code(prompt, top_k=3):
    query_vec = embedder.encode([prompt])[0].tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    return list(set(results["documents"][0]))

# Generate code or output using retrieved chunks
def generate_code(user_prompt, retrieved_chunks, max_input_tokens=2048, max_new_tokens=512):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant.

Use the following context from the project:

{context}

Based on the above, perform this instruction:
\"\"\"
{user_prompt}
\"\"\"
"""

    # Tokenize input safely
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)

    # Move model and inputs to CPU (avoid Meta tensor issues)
    input_ids = inputs["input_ids"].to("cpu")
    attention_mask = inputs["attention_mask"].to("cpu")
    model.to("cpu")  # Ensure model is not in 'meta' state

    # Generate safely
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove input prompt from generated output
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text.replace(input_text, "").strip()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Code Assistant", layout="wide")
st.title("RAG-Powered Code Generator")
st.markdown("Give a natural prompt and get code based on your project context!")

with st.sidebar:
    st.header("📦 Codebase Loader")
    if st.button("🔄 Load & Embed Codebase"):
        load_and_embed_code()
        st.success("✅ Codebase embedded into Chroma DB.")

prompt = st.text_area("💬 Enter your instruction or code generation request:", height=150)

if st.button("🚀 Generate Output"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        st.info("⏳ Thinking...")
        relevant_chunks = retrieve_code(prompt)
        if not relevant_chunks:
            st.warning("⚠️ No relevant context found.")
        else:
            generated_code = generate_code(prompt, relevant_chunks)
            st.success("✅ Output Generated!")
            st.code(generated_code, language="python")
