import streamlit as st
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch

# ---- PAGE SETUP ----
st.set_page_config(page_title="🗞️ HW7 - News Info Bot", layout="wide")
st.title("🗞️ HW7 - News Info Bot | Law Firm Edition")
st.write("Ask questions about the uploaded news data — powered by RAG + LLM comparison.")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv("HW7_news.csv")
    df.dropna(subset=["Document"], inplace=True)
    return df

df = load_data()
st.subheader("Preview of News Data")
st.dataframe(df.head())

# ---- EMBEDDING MODEL ----
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# Create embeddings for all documents
if "embeddings" not in st.session_state:
    with st.spinner("Encoding news articles..."):
        st.session_state.embeddings = embedder.encode(df["Document"].tolist(), convert_to_tensor=True)

# ---- LLM SETUP ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)

if not OPENAI_API_KEY:
    st.warning("⚠️ Please add your OpenAI API key to .streamlit/secrets.toml")

# ---- USER INPUT ----
user_query = st.text_input("Enter your query (e.g., 'find the most interesting news' or 'find news about JPMorgan')")

# ---- HELPER FUNCTIONS ----
def retrieve_relevant_news(query, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, st.session_state.embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    return [(df.iloc[idx.item()], scores[idx].item()) for idx in top_results.indices]

def summarize_with_model(vendor, model, context, query):
    if vendor == "OpenAI":
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"You are a news assistant for a global law firm. Based on the following articles, answer the user's query: {query}\n\n{context}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a precise, factual legal news assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    return "Vendor not supported yet."

# ---- PROCESS QUERY ----
if user_query:
    st.subheader("Results")
    results = retrieve_relevant_news(user_query, top_k=5)
    
    if "interesting" in user_query.lower():
        st.markdown("### 🔝 Most Interesting News (by semantic similarity)")
    else:
        st.markdown("### 🔎 Relevant News Articles")

    for i, (row, score) in enumerate(results):
        st.markdown(f"**{i+1}. [{row['Document'][:80]}...]({row['URL']})**")
        st.caption(f"{row['company_name']} | {row['Date']} | Similarity: {score:.4f}")

    # ---- RAG CONTEXT ----
    combined_context = "\n\n".join([r[0]['Document'] for r in results])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💬 OpenAI GPT-4 (expensive model)")
        if OPENAI_API_KEY:
            with st.spinner("Querying OpenAI GPT-4..."):
                answer_gpt4 = summarize_with_model("OpenAI", "gpt-4-turbo", combined_context, user_query)
                st.write(answer_gpt4)
        else:
            st.info("Add your OpenAI API key in `.streamlit/secrets.toml` to use this model.")
    with col2:
        st.markdown("#### 💬 OpenAI GPT-3.5 (cheaper model)")
        if OPENAI_API_KEY:
            with st.spinner("Querying OpenAI GPT-3.5..."):
                answer_gpt35 = summarize_with_model("OpenAI", "gpt-3.5-turbo", combined_context, user_query)
                st.write(answer_gpt35)

# ---- EXPLANATION SECTION ----
with st.expander("🧠 Architecture & Evaluation"):
    st.markdown("""
    **Architecture:**
    - CSV file → Embedded using SentenceTransformer (MiniLM) for efficient semantic search.
    - RAG pipeline retrieves top-5 relevant articles for a query.
    - Query and context sent to two LLMs (OpenAI GPT-4 & GPT-3.5) for comparison.
    
    **Ranking Quality:**
    - Measured via semantic similarity using cosine scores between user query and document embeddings.
    - Manual validation done by checking if top-ranked articles are contextually relevant to user queries.

    **Model Comparison:**
    - GPT-4 gives more nuanced, detailed, and legally contextualized responses.
    - GPT-3.5 is faster and cheaper, sufficient for basic summarization and topic filtering.
    """)

