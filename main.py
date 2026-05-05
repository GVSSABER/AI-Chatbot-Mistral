import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI RAG Chatbot")
st.write("Upload PDF and ask questions")

# -------------------------
# PDF UPLOAD
# -------------------------
st.header("📄 Upload PDF")

uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}

    with st.spinner("Uploading..."):
        response = requests.post(f"{API_URL}/upload", files={"file": uploaded_file})

    st.success("Uploaded Successfully")
    st.json(response.json())

# -------------------------
# CHAT SECTION
# -------------------------
st.header("💬 Ask Questions")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if question:

        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={"question": question}
            )

        data = response.json()

        st.subheader("Answer:")
        st.write(data.get("answer", "No answer"))

        st.subheader("Relevant Context:")
        st.write(data.get("relevant_chunks", []))
