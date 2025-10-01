import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# Function to read PDF files
def read_pdf(uploaded_file):
    text = ""
    file_bytes = uploaded_file.read()  # store bytes to avoid stream exhaustion
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


st.title("🎈FileBot - HW1 - Bhushan Jain ")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "Provide your OpenAI API key."
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    if uploaded_file is None and "document" in st.session_state:
        del st.session_state["document"]

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            st.session_state["document"] = uploaded_file.read().decode()
        elif file_extension == "pdf":
            st.session_state["document"] = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

    disabled = "document" not in st.session_state
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled=disabled
    )

    models = {
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o"
    }

    if "document" in st.session_state and question:
        for label, mid in models.items():
            st.subheader(f"🤖 Model: {label}")
            try:
                response = client.chat.completions.create(
                    model=mid,
                    messages=[
                        {"role": "system", "content": "Answer using only the document provided."},
                        {"role": "user", "content": f"Document: {st.session_state['document']}\n\nQuestion: {question}"}
                    ],
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"{label} failed: {e}")
