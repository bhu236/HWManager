# HWs/HW3.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import textwrap
import json

# Optional vendor SDK imports (wrapped so missing packages won't crash the app)
try:
    import anthropic
    ANTHROPIC_INSTALLED = True
except Exception:
    ANTHROPIC_INSTALLED = False

# ---------- CONFIG ----------
MAX_TOKEN_BUFFER_DEFAULT = 2000  # used when user selects 'buffer of 2000 tokens'
STREAM_CHUNK_SIZE = 80  # characters per UI update for non-streaming vendors
# ----------------------------

# Simple token estimation (word-based proxy)
def count_tokens(messages):
    return sum(len(m["content"].split()) for m in messages)

# Read text from URL
def read_url_content(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        st.warning(f"Couldn't read URL {url}: {e}")
        return ""

def system_prompt_for_kid():
    return {
        "role": "system",
        "content": (
            "You are a helpful, friendly assistant that explains things clearly "
            "so a 10-year-old can understand. Use simple words, short sentences, "
            "and examples or analogies when helpful."
        ),
    }

def prepare_messages(url_texts, session_messages, memory_mode, memory_param):
    messages = [system_prompt_for_kid()]
    for idx, txt in enumerate(url_texts, start=1):
        if txt:
            snippet = txt[:4000]
            messages.append({
                "role": "system",
                "content": f"Source Document {idx}:\n{snippet}\n\n(End of source {idx})"
            })
    if memory_mode == "buffer_6":
        user_msgs = [m for m in session_messages if m["role"] == "user"][-6:]
        assistant_msgs = [m for m in session_messages if m["role"] == "assistant"][-6:]
        buffer = []
        for i in range(len(user_msgs)):
            buffer.append(user_msgs[i])
            if i < len(assistant_msgs):
                buffer.append(assistant_msgs[i])
        messages.extend(buffer)
    elif memory_mode == "conversation_summary":
        if session_messages:
            convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in session_messages[-50:]])
            messages.append({"role": "system", "content": "Previous conversation (for context):\n" + convo_text})
            messages.append({"role": "system", "content": "From the previous conversation above, consider the key points as context when answering."})
    elif memory_mode == "buffer_tokens":
        recent = []
        total = 0
        for m in reversed(session_messages):
            m_tokens = len(m["content"].split())
            if total + m_tokens > memory_param:
                break
            recent.insert(0, m)
            total += m_tokens
        messages.extend(recent)
    return messages

# Streaming helpers
def stream_response_openai(openai_client, model_id, messages):
    stream = openai_client.chat.completions.create(model=model_id, messages=messages, stream=True)
    for event in stream:
        delta = event.choices[0].delta
        content = None
        if isinstance(delta, dict):
            content = delta.get("content")
        else:
            content = getattr(delta, "content", None)
        if content:
            yield content

def stream_response_other(text, chunk_size=STREAM_CHUNK_SIZE):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

# ----------------- Vendor Sync Calls -----------------
def call_deepseek_sync(deepseek_key, model, prompt_text):
    """
    Synchronous call to Deepseek API.
    """
    try:
        url = "https://api.deepseek.ai/v1/generate"
        headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": 512
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        return resp.json().get("text", "")
    except Exception as e:
        return f"(Deepseek API error: {e})"

def call_anthropic_sync(anthropic_key, model, prompt_text):
    if not ANTHROPIC_INSTALLED:
        return "Anthropic SDK not installed on server."
    client = anthropic.Client(api_key=anthropic_key)
    resp = client.completions.create(model=model, prompt=prompt_text, max_tokens_to_sample=512)
    return resp.completion

# ----------------- STREAMLIT UI -----------------
def app():
    st.title("HW3 — URL-based Streaming Chatbot (Multi-LLM)")

    st.sidebar.header("URL Inputs")
    url1 = st.sidebar.text_input("URL #1 (required)", value="https://www.howbaseballworks.com/TheBasics.htm")
    url2 = st.sidebar.text_input("URL #2 (optional)", value="https://www.pbs.org/kenburns/baseball/baseball-for-beginners")

    st.sidebar.markdown("---")
    st.sidebar.header("Select LLM Vendor & Model")
    vendor = st.sidebar.selectbox("Choose vendor", ["OpenAI", "Deepseek", "Anthropic"])
    vendor_model_map = {
        "OpenAI": ["gpt-4o-mini (cheap)", "gpt-4o (flagship)"],
        "Deepseek": ["ds-mini", "ds-large"],
        "Anthropic": ["claude-2.1 (cheap)", "claude-2.1-100k (flagship)"]
    }
    model_choice = st.sidebar.selectbox("Model", vendor_model_map.get(vendor, []))

    st.sidebar.markdown("---")
    st.sidebar.header("Conversation Memory Mode")
    memory_mode = st.sidebar.selectbox("Memory type", ["buffer_6", "conversation_summary", "buffer_tokens"])
    memory_param = st.sidebar.number_input("Max tokens for buffer (word-proxy)", min_value=200, max_value=5000, value=2000, step=100) if memory_mode=="buffer_tokens" else None

    st.sidebar.markdown("---")
    st.sidebar.caption("Provide URLs; the system will use them as context.")

    if st.button("Fetch & Preview URLs"):
        st.info("Fetching URLs — may take a few seconds.")
        url_text1 = read_url_content(url1) if url1 else ""
        url_text2 = read_url_content(url2) if url2 else ""
        if url_text1:
            st.subheader("Preview URL #1")
            st.write(textwrap.shorten(url_text1, width=1000, placeholder="..."))
        if url_text2:
            st.subheader("Preview URL #2")
            st.write(textwrap.shorten(url_text2, width=1000, placeholder="..."))

    # API keys
    openai_client = None
    if vendor=="OpenAI":
        try:
            openai_key = st.secrets["OPENAI_API_KEY"]
            openai_client = OpenAI(api_key=openai_key)
        except Exception:
            st.error("OpenAI API key not found.")
            return
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") if ANTHROPIC_INSTALLED else None

    # Session state
    if "hw3_messages" not in st.session_state:
        st.session_state.hw3_messages = []
    if "hw3_awaiting_more_info" not in st.session_state:
        st.session_state.hw3_awaiting_more_info = False

    for msg in st.session_state.hw3_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("Ask a question about the provided URLs (or general):"):
        st.session_state.hw3_messages.append({"role": "user", "content": user_q})

        url_texts = [read_url_content(url1) if url1 else ""]
        if url2:
            url_texts.append(read_url_content(url2))

        if st.session_state.hw3_awaiting_more_info:
            follow_prompt = "Provide more explanation in simple language for a 10-year-old. Then ask: 'Do you want more info?'" if user_q.strip().lower() in ["yes","y"] else "Ask the user: 'What else can I help you with?'"
            st.session_state.hw3_awaiting_more_info = user_q.strip().lower() in ["yes","y"]
            final_user_prompt = follow_prompt
        else:
            final_user_prompt = f"Answer this question in simple language for a 10-year-old: {user_q}\nThen ask 'Do you want more info?'"
            st.session_state.hw3_awaiting_more_info = True

        messages = prepare_messages(url_texts, st.session_state.hw3_messages, memory_mode, memory_param or MAX_TOKEN_BUFFER_DEFAULT)
        messages_to_send = messages + [{"role": "user", "content": final_user_prompt}]

        while count_tokens(messages_to_send) > (memory_param if memory_param else MAX_TOKEN_BUFFER_DEFAULT) and len(messages_to_send) > 2:
            idx_to_pop = next((i for i, m in enumerate(messages_to_send) if m.get("role")!="system"), None)
            if idx_to_pop is None:
                break
            messages_to_send.pop(idx_to_pop)

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            collected = ""

            try:
                if vendor=="OpenAI" and openai_client:
                    model_id = "gpt-4o" if "flagship" in model_choice or "4o" in model_choice else "gpt-4o-mini"
                    for chunk in stream_response_openai(openai_client, model_id, messages_to_send):
                        collected += chunk
                        placeholder.markdown(collected + "▌")
                elif vendor=="Deepseek":
                    if not deepseek_key:
                        placeholder.error("Deepseek API key missing.")
                    else:
                        combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
                        ds_text = call_deepseek_sync(deepseek_key, model_choice, combined_prompt)
                        for chunk in stream_response_other(ds_text):
                            collected += chunk
                            placeholder.markdown(collected + "▌")
                elif vendor=="Anthropic":
                    if not ANTHROPIC_INSTALLED or not anthropic_key:
                        placeholder.error("Anthropic not available: SDK missing or API key missing.")
                    else:
                        combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
                        anth_text = call_anthropic_sync(anthropic_key, model_choice, combined_prompt)
                        for chunk in stream_response_other(anth_text):
                            collected += chunk
                            placeholder.markdown(collected + "▌")
                else:
                    placeholder.error("Unsupported vendor or missing API key.")
                    collected = ""
                placeholder.markdown(collected)
            except Exception as e:
                placeholder.error(f"Error generating response: {e}")
                collected = f"(error: {e})"

        st.session_state.hw3_messages.append({"role": "assistant", "content": collected})
        approx_tokens = count_tokens(messages_to_send) + len(collected.split())
        st.info(f"Approx. tokens sent to model (word-proxy): {approx_tokens}")

    st.markdown("---")
    st.subheader("HW3 Evaluation Instructions")
    st.markdown(
        """
        Use the two default URLs about baseball to evaluate:
        - https://www.howbaseballworks.com/TheBasics.htm
        - https://www.pbs.org/kenburns/baseball/baseball-for-beginners

        For HW submission:
        1. Define 3 evaluation questions.
        2. Run chatbot in 6 scenarios with vendors & URLs.
        3. Compare outputs and decide best model & URL usage.
        """
    )

if __name__ == "__main__":
    app()
