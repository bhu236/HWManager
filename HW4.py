# HW4.py
# iSchool Chatbot using RAG
# Author: Bhushan Jain
# Goal: Chatbot that answers questions about iSchool student organizations

import os
import sys
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# 1) ENVIRONMENT / SQLITE FIX (must run BEFORE importing chromadb internals)
# ---------------------------------------------------------------------
try:
    __import__("pysqlite3")
    # replace stdlib sqlite3 with pysqlite3 to avoid Streamlit cloud sqlite issues
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # not fatal — warn the user
    st.warning("pysqlite3 not found. ChromaDB may fail on platforms with old sqlite3. "
               "Install pysqlite3 in requirements if you see sqlite errors.")

# ---------------------------------------------------------------------
# 2) ChromaDB persistent client
# ---------------------------------------------------------------------
import chromadb
from chromadb.api.models.Collection import Collection  # type: ignore

# ---------------------------------------------------------------------
# 3) CONFIG
# ---------------------------------------------------------------------
CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "html_docs"  # folder containing the HTML files
CHROMA_COLLECTION_NAME = "MultiDocCollection"

MODEL_OPTIONS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Google": ["gemini-1.5", "gemini-1.5-pro"],
    "Anthropic": ["claude-3.5-sonnet", "claude-3-haiku"]
}

# ---------------------------------------------------------------------
# 4) CACHED RESOURCES
# ---------------------------------------------------------------------
@st.cache_resource
def get_api_clients():
    """
    Initialize and cache API clients. Keys should be stored in Streamlit secrets.
    Exits cleanly (st.stop) if any required key is missing.
    """
    try:
        # configure google generative ai if key present
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

        clients = {
            "OpenAI": OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None)),
            "Anthropic": Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY", None)),
            # "Google" will use `genai` module directly
        }
        return clients
    except Exception as e:
        st.error(f"Failed to initialize API clients. Check secrets. Error: {e}")
        st.stop()


@st.cache_resource
def get_chroma_collection():
    """
    Create or open a persistent Chroma collection.
    Uses chromadb.PersistentClient to avoid deprecated Settings usage.
    """
    persist_path = Path(CHROMA_DB_PATH)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    # get_or_create_collection returns a collection object
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return client, collection


# ---------------------------------------------------------------------
# 5) UTILITIES: read HTML, chunk, embed, query
# ---------------------------------------------------------------------
def extract_text_from_html(file_path: str) -> str:
    """Extract visible text from HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            # remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            # collapse multiple blank lines
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return ""


def setup_vector_db(collection: Collection, openai_client: OpenAI, force_rebuild: bool = False):
    """
    Populate the Chroma collection from HTML documents in SOURCE_DIR.
    Uses RecursiveCharacterTextSplitter to create chunks from each doc.
    Runs only when collection is empty or force_rebuild is True.
    """
    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB already has {existing_count} chunks.")
        return

    st.sidebar.warning("Building Vector DB from HTML files (may take a minute)...")
    html_folder = Path(SOURCE_DIR)
    if not html_folder.exists():
        st.sidebar.error(f"Source folder '{SOURCE_DIR}' does not exist. Create it and add HTML files.")
        st.stop()

    files = sorted([p for p in html_folder.glob("*.html")])
    if not files:
        st.sidebar.error(f"No HTML files found in '{SOURCE_DIR}'. Place your HTML files there.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs, ids, metadatas = [], [], []

    for file in files:
        text = extract_text_from_html(str(file))
        if not text:
            continue

        # Create two mini-documents per file if document is small; otherwise chunk normally
        # Explanation: We want two mini-documents per the HW requirement for small files,
        # but longer files get chunked into multiple chunks to preserve context.
        if len(text) < 2500:
            # try to split roughly in half on sentence boundary
            mid = len(text) // 2
            # find nearest newline from mid to avoid breaking sentence
            split_idx = text.rfind("\n", 0, mid)
            if split_idx <= 0:
                split_idx = mid
            chunk1 = text[:split_idx].strip()
            chunk2 = text[split_idx:].strip()
            chunks = [c for c in (chunk1, chunk2) if c]
        else:
            chunks = text_splitter.split_text(text)

        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{file.stem}_chunk{i+1}")
            metadatas.append({"source": file.name, "chunk": i+1})

    # Batch embed and add to collection
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        try:
            emb_resp = openai_client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            embeddings = [r.embedding for r in emb_resp.data]
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta, embeddings=embeddings)
        except Exception as e:
            st.error(f"Failed to embed/add batch starting at {i}: {e}")
            # continue trying remaining batches

    # Persist (PersistentClient handles persistence automatically)
    final_count = collection.count()
    st.sidebar.success(f"Vector DB built with {final_count} chunks.", icon="✅")


def query_vector_db(collection: Collection, openai_client: OpenAI, prompt: str, n_results: int = 4) -> str:
    """Return joined document text from top n_results nearest neighbors."""
    try:
        q_emb = openai_client.embeddings.create(input=[prompt], model="text-embedding-3-small").data[0].embedding
        results = collection.query(query_embeddings=[q_emb], n_results=n_results, include=["documents", "metadatas", "distances"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return "No relevant context found."
        parts = []
        for d, m in zip(docs, metas):
            src = m.get("source", "unknown")
            chunk_idx = m.get("chunk", "?")
            parts.append(f"Source: {src} (chunk {chunk_idx})\n{d}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        st.error(f"Vector DB query failed: {e}")
        return "Error retrieving context."


# ---------------------------------------------------------------------
# 6) LLM orchestration
# ---------------------------------------------------------------------
def get_llm_response(clients: dict, provider: str, model_name: str, question: str, context: str, chat_history: str) -> str:
    """
    Ask selected provider/model to answer using context + history.
    This function attempts to adapt calls for different providers; adapt keys/models as needed.
    """
    system_prompt = "You are an expert assistant. Use the context to answer the question. If the answer is not in the context, say you cannot find it and give guidance."
    user_prompt = f"CONTEXT:\n{context}\n\nHISTORY:\n{chat_history}\n\nQUESTION:\n{question}"

    try:
        with st.spinner(f"Asking {provider} ({model_name})..."):
            if provider == "OpenAI":
                client = clients["OpenAI"]
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    max_tokens=800
                )
                return resp.choices[0].message.content

            elif provider == "Google":
                # use genai (configured in get_api_clients)
                model = genai.Model(model=model_name)
                result = model.generate_text(prompt=f"{system_prompt}\n\n{user_prompt}")
                return result.text if hasattr(result, "text") else str(result)

            elif provider == "Anthropic":
                client = clients["Anthropic"]
                # Anthropic API wrapper interface may vary — adapt as needed for your Anthropic version
                resp = client.messages.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    max_tokens_to_sample=800
                )
                # attempt to extract text
                if hasattr(resp, "content") and isinstance(resp.content, list) and resp.content:
                    return resp.content[0].get("text", str(resp))
                # fallback
                return str(resp)
    except Exception as e:
        st.error(f"Error with {provider}/{model_name}: {e}")
        return "Sorry — model request failed."


# ---------------------------------------------------------------------
# 7) STREAMLIT UI / APP FLOW
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multi-LLM RAG Chat", page_icon="🧠")
    st.title("🧠 Multi-LLM RAG iSchool ChatBot Application")
    st.write(f"Documents read from folder: `{SOURCE_DIR}`")

    # init clients & collection
    clients = get_api_clients()
    chroma_client, collection = get_chroma_collection()

    # Sidebar controls
    st.sidebar.header("Model / DB Settings")
    selected_provider = st.sidebar.selectbox("LLM Provider", options=list(MODEL_OPTIONS.keys()))
    selected_model = st.sidebar.selectbox("Model", options=MODEL_OPTIONS[selected_provider])

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear chat history"):
        st.session_state.pop("messages", None)
        st.experimental_rerun()

    if st.sidebar.button("Rebuild Vector DB"):
        # delete collection and rebuild
        try:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        # clear cached collection resource so get_chroma_collection recreates
        get_chroma_collection.clear()
        st.experimental_rerun()

    # Build vector DB if needed (OpenAI client passed for embeddings)
    setup_vector_db(collection, clients["OpenAI"], force_rebuild=False)

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # display existing messages
    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    # get user input
    user_prompt = st.chat_input("Ask a question about iSchool student organizations...")
    if user_prompt:
        # append user message
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # retrieve context
        context = query_vector_db(collection, clients["OpenAI"], user_prompt, n_results=4)
        # assemble short chat history (last 10)
        recent_msgs = st.session_state["messages"][-10:]
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

        # get response from selected LLM
        answer = get_llm_response(clients, selected_provider, selected_model, user_prompt, context, chat_history)

        full_answer = f"**Answer from {selected_provider} / {selected_model}:**\n\n{answer}"
        st.session_state["messages"].append({"role": "assistant", "content": full_answer})
        with st.chat_message("assistant"):
            st.markdown(full_answer)


if __name__ == "__main__":
    main()
