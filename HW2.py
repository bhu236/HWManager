import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
# Placeholder for other LLMs:
# from cohere import Cohere
# from anthropic import Anthropic

st.title("🌐 URL Summarizer (HW2)")

# --- URL input ---
url = st.text_input("Enter a URL to summarize:")

# --- Sidebar ---
summary_type = st.sidebar.selectbox(
    "📝 Type of Summary",
    ["Concise Overview", "Detailed Summary", "Bullet Points", "Key Insights"]
)

language = st.sidebar.selectbox(
    "🌐 Output Language",
    ["English", "French", "Spanish"]
)

llm_choice = st.sidebar.selectbox(
    "🤖 Select LLM",
    ["OpenAI GPT", "Claude", "Cohere"]
)

use_advanced_model = st.sidebar.checkbox("Use Advanced Model", value=True)

# --- Function to read URL content ---
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# --- Function to summarize text using selected LLM ---
def summarize_text(text, llm, summary_type, language):
    prompt = f"Summarize the following text into {summary_type} in {language}:\n\n{text}"

    if llm == "OpenAI GPT":
        client = OpenAI(api_key=st.secrets["openai_api_key"])
        model_name = "gpt-4o-mini" if use_advanced_model else "gpt-3.5-turbo"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summarizing documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        return response.choices[0].message.content

    elif llm == "Cohere":
        # Placeholder: integrate cohere API here
        return "Cohere summary placeholder."

    elif llm == "Claude":
        # Placeholder: integrate claude/anthropic API here
        return "Claude summary placeholder."

    else:
        return "Selected LLM not supported."

# --- Main ---
if url:
    with st.spinner("📖 Reading URL content..."):
        text = read_url_content(url)

    if text:
        st.success("✅ URL content extracted!")

        if st.button("Generate Summary"):
            with st.spinner("✨ Generating summary..."):
                summary = summarize_text(text, llm_choice, summary_type, language)
                st.subheader("📌 Summary")
                st.write(summary)
