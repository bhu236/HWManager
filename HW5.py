# HW5.py
# Short-term memory RAG Chatbot with Multi-LLM Support
# Author: Bhushan Jain (adapted)

import os
import sys
from pathlib import Path
import streamlit as st
from bs4 import BeautifulSoup

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

import chromadb
from chromadb.api.models.Collection import Collection

from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------- CONFIG --------------------
CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "html_docs"
CHROMA_COLLECTION_NAME = "MultiDocCollection"

MODEL_OPTIONS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Google": ["gemini-1.5", "gemini-1.5-pro"],
    "Anthropic": ["claude-3.5-sonnet", "claude-3-haiku"]
}

N_MEMORY = 6  # number of past turns stored

# -------------------- HELPERS --------------------
def extract_text_from_html(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return ""


# -------------------- CACHE RESOURCES --------------------
@st.cache_resource
def get_api_clients():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        return {
            "OpenAI": OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None)),
            "Anthropic": Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY", None))
        }
    except Exception as e:
        st.error(f"Failed to initialize API clients: {e}")
        st.stop()


@st.cache_resource
def get_chroma_collection():
    persist_path = Path(CHROMA_DB_PATH)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return client, collection


# -------------------- VECTOR DB --------------------
def setup_vector_db(collection: Collection, openai_client: OpenAI, force_rebuild: bool = False):
    existing_count = collection.count() if hasattr(collection, "count") else 0
    if existing_count > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB already has {existing_count} chunks.")
        return

    st.sidebar.warning("Building Vector DB from HTML files...")
    html_folder = Path(SOURCE_DIR)
    if not html_folder.exists():
        st.sidebar.error(f"Source folder '{SOURCE_DIR}' does not exist.")
        st.stop()

    files = sorted([p for p in html_folder.glob("*.html")])
    if not files:
        st.sidebar.error(f"No HTML files found in '{SOURCE_DIR}'.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs, ids, metadatas = [], [], []

    for file in files:
        text = extract_text_from_html(str(file))
        if not text:
            continue
        chunks = (
            [text[:len(text)//2].strip(), text[len(text)//2:].strip()]
            if len(text) < 2500 else text_splitter.split_text(text)
        )
        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{file.stem}_chunk{i+1}")
            metadatas.append({"source": file.name, "chunk": i+1})

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        try:
            emb_resp = openai_client.embeddings.create(
                input=batch_docs, model="text-embedding-3-small"
            )
            embeddings = [r.embedding for r in emb_resp.data]
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta, embeddings=embeddings)
        except Exception as e:
            st.error(f"Failed to embed batch starting at {i}: {e}")

    st.sidebar.success(f"Vector DB built with {collection.count()} chunks.")


def query_vector_db_return_docs(collection: Collection, openai_client: OpenAI, prompt: str, n_results: int = 4):
    try:
        q_emb = openai_client.embeddings.create(
            input=[prompt], model="text-embedding-3-small"
        ).data[0].embedding
        results = collection.query(query_embeddings=[q_emb], n_results=n_results, include=["documents", "metadatas"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [(d, m) for d, m in zip(docs, metas)] if docs else []
    except Exception as e:
        st.error(f"Vector DB query failed: {e}")
        return []


def get_relevant_course_info(query: str, collection: Collection, openai_client: OpenAI, n_results: int = 4):
    hits = query_vector_db_return_docs(collection, openai_client, query, n_results)
    if not hits:
        return "No relevant context found."
    return "\n\n---\n\n".join(
        [f"Source: {m.get('source', 'unknown')} (chunk {m.get('chunk', '?')})\n{d}" for d, m in hits]
    )


# -------------------- LLM ORCHESTRATION --------------------
def get_llm_response(clients: dict, provider: str, model_name: str, question: str, retrieved_context: str, short_memory: str):
    system_prompt = (
        "You are an expert assistant. Use ONLY the retrieved context to answer the question."
        " If context is insufficient, say so. Cite sources."
    )
    user_prompt = (
        f"RETRIEVED_CONTEXT:\n{retrieved_context}\n\n"
        f"SHORT_TERM_MEMORY:\n{short_memory}\n\n"
        f"QUESTION:\n{question}\n\n"
    )

    try:
        if provider == "OpenAI":
            client = clients["OpenAI"]
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=800
            )
            return resp.choices[0].message.content
        elif provider == "Google":
            model = genai.Model(model=model_name)
            result = model.generate_text(prompt=f"{system_prompt}\n\n{user_prompt}")
            return getattr(result, "text", str(result))
        elif provider == "Anthropic":
            client = clients["Anthropic"]
            resp = client.messages.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens_to_sample=800
            )
            if hasattr(resp, "content") and resp.content:
                return resp.content[0].get("text", str(resp))
            return str(resp)
    except Exception as e:
        st.error(f"Error with {provider}/{model_name}: {e}")
        return "Sorry — model request failed."


# -------------------- STREAMLIT APP --------------------
def run():
    st.set_page_config(page_title="HW5 - Short-Term Memory Chatbot", layout="wide")
    st.title("HW5 - Short-Term Memory Doc Bot with Multi-LLM support")

    clients = get_api_clients()
    chroma_client, collection = get_chroma_collection()

    provider = st.sidebar.selectbox("LLM Provider", list(MODEL_OPTIONS.keys()))
    model_name = st.sidebar.selectbox("Model", MODEL_OPTIONS[provider])

    if st.sidebar.button("Clear Chat History"):
        st.session_state.pop("messages", None)
        st.session_state.pop("short_term_memory", None)
        st.experimental_rerun()

    if st.sidebar.button("Rebuild Vector DB"):
        try:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        get_chroma_collection.clear()
        setup_vector_db(collection, clients["OpenAI"], force_rebuild=True)
        st.experimental_rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "short_term_memory" not in st.session_state:
        st.session_state["short_term_memory"] = []

    user_input = st.chat_input("Ask your question about the documents...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["short_term_memory"].append(f"User: {user_input}")
        if len(st.session_state["short_term_memory"]) > N_MEMORY:
            st.session_state["short_term_memory"] = st.session_state["short_term_memory"][-N_MEMORY:]

        retrieved_context = get_relevant_course_info(user_input, collection, clients["OpenAI"], n_results=4)
        short_memory_text = "\n".join(st.session_state["short_term_memory"])

        answer = get_llm_response(clients, provider, model_name, user_input, retrieved_context, short_memory_text)

        formatted_answer = f"**Answer ({provider}/{model_name})**\n\n{answer}\n\n---\n\n**Retrieved Context:**\n{retrieved_context}"
        st.session_state["messages"].append({"role": "assistant", "content": formatted_answer})
        st.session_state["short_term_memory"].append(f"Assistant: {answer}")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if __name__ == "__main__":
    run()
