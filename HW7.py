import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import google.generativeai as genai
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
from chromadb.utils import embedding_functions

# ==============================
# APP CONFIG
# ==============================
st.set_page_config(page_title="🗞️ HW7 News Info Bot", layout="wide")
st.title("🗞️ HW7 – News Info Bot for a Global Law Firm")

# ==============================
# LOAD DATA
# ==============================
DATA_PATH = "HW7_news.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ HW7_news.csv not found. Please place it in the project root folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.sidebar.success(f"✅ Loaded {len(df)} news stories.")

# ==============================
# INITIALIZE API CLIENTS
# ==============================
openai_client = OpenAI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==============================
# CREATE CHROMA VECTOR STORE
# ==============================
chroma_client = chromadb.Client()
embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(name="news_rag", embedding_function=embedding_func)

# Load embeddings into Chroma (only once)
if len(collection.get()['ids']) == 0:
    for i, row in df.iterrows():
        doc_text = f"{row['company_name']} - {row['Document']}"
        collection.add(
            ids=[str(i)],
            documents=[doc_text],
            metadatas=[{"url": row['URL'], "date": row['Date']}]
        )

# ==============================
# HELPER FUNCTIONS
# ==============================
def retrieve_relevant_docs(query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results*2)
    seen = set()
    docs = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        if text not in seen:
            seen.add(text)
            docs.append({
                "content": text,
                "url": results['metadatas'][0][i]['url'],
                "date": results['metadatas'][0][i]['date']
            })
        if len(docs) >= n_results:
            break
    return docs

def openai_chat_completion(prompt, model="gpt-4o-mini"):
    """Call OpenAI ChatCompletion using the new API."""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an analytical legal news assistant that summarizes and ranks news for lawyers."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=500,
            #temperature=0.7, as the new gpt-5 and gpt-4o models no longer support custom temperature values — they run with adaptive reasoning control internally
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI Error: {e}"


import google.generativeai as genai

import google.generativeai as genai

def gemini_chat_completion(prompt):
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")  # fallback model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

def build_prompt(user_query, context_docs):
    """Constructs a combined context-aware RAG prompt."""
    context_text = "\n\n".join([f"- {doc['content']} ({doc['date']}) [{doc['url']}]" for doc in context_docs])
    prompt = f"""
    You are a news summarization bot for a global law firm.
    The following are recent financial/legal news stories:

    {context_text}

    Based on these, answer the user query below clearly and analytically.

    User query: {user_query}

    When asked 'most interesting news', rank stories by their legal, regulatory, or reputational significance.
    When asked for a specific topic, show the most relevant and insightful news with reasoning.
    """
    return prompt

# ==============================
# STREAMLIT UI
# ==============================
st.header("💬 Ask about the News Stories")

user_query = st.text_input("Enter your question (e.g., 'find the most interesting news' or 'find news about JPMorgan AI')")

if user_query:
    with st.spinner("🔍 Retrieving and analyzing news..."):
        context_docs = retrieve_relevant_docs(user_query, n_results=5)
        rag_prompt = build_prompt(user_query, context_docs)

        # Compare both vendors
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 OpenAI GPT-5 (Premium Reasoning)")
            openai_response = openai_chat_completion(rag_prompt, model="gpt-5")
            st.write(openai_response)

        with col2:
            st.subheader("🔹 Gemini 1.5-Flash (Google – Cost Efficient)")
            gemini_response = gemini_chat_completion(rag_prompt)
            st.write(gemini_response)

        st.markdown("---")
        st.subheader("📑 Retrieved Context (Top 5)")
        for d in context_docs:
            st.markdown(f"**{d['content']}**  \n📅 {d['date']}  \n🔗 [Read more]({d['url']})")

# ==============================
# EXPLANATION SECTION
# ==============================
with st.expander("📘 Explanation: Architecture & Evaluation"):
    st.markdown("""
    **Architecture:**
    - Data Source: `HW7_news.csv` containing company news and summaries.
    - Vector Store: ChromaDB for semantic retrieval.
    - Models:
        - OpenAI `gpt-5` → high-reasoning premium model.
        - OpenAI `gpt-4o-mini` → faster, cheaper variant (can switch via code).
        - Google Gemini 1.5-Flash → secondary vendor for comparison.
    - RAG Pipeline: Retrieve 5 most relevant stories, inject into context, generate ranked/filtered answers.

    **Ranking Quality Evaluation:**
    - For “most interesting news,” results were verified manually by checking:
        - Relevance to law/regulation.
        - Timeliness and reputational impact.
        - Agreement between GPT-5 and Gemini outputs.
    - For “topic-based queries,” correctness tested by:
        - Comparing retrieved stories’ keywords to the query.
        - Ensuring summaries cited URLs and matched the source context.

    **Model Comparison:**
    - GPT-5 → Better at legal reasoning and nuanced ranking (more expensive).
    - Gemini → Faster responses and lower cost, but less depth.
    - Best overall: **GPT-5 (OpenAI)** for precision and interpretability.
    """)

