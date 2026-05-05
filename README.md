# AI-Chatbot-Mistral
This project is an AI-powered chatbot that can answer questions based on uploaded PDF documents. It uses Retrieval-Augmented Generation (RAG) with embeddings and a vector store to provide accurate, context-based answers.
# 🤖 AI Chatbot with PDF RAG (Mistral 7B)

## 📌 Overview

This project is an AI-powered chatbot that can answer questions based on uploaded PDF documents.
It uses **Retrieval-Augmented Generation (RAG)** with embeddings and a vector store to provide accurate, context-based answers.

---

## 🚀 Features

* 📄 Upload PDF files
* 🧠 Extract and chunk text
* 🔎 Semantic search using embeddings
* 🤖 AI responses using Mistral-7B
* ⚡ FastAPI backend
* 📊 Context-aware answers (RAG)

---

## 🛠️ Tech Stack

* FastAPI
* Transformers (Hugging Face)
* Sentence Transformers
* PyTorch
* Python

---

## 📂 Project Structure

```
backend/
  ├── main.py
  ├── services/
       ├── pdf_loader.py
       ├── chunker.py
       ├── embeddings.py
       ├── vector_store.py

data/
  ├── pdfs/

requirements.txt
README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ai-chatbot.git
cd ai-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
uvicorn backend.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

## 📤 API Endpoints

### Upload PDF

```
POST /upload
```

### Chat with AI

```
POST /chat
```

#### Example Request:

```json
{
  "question": "What is Artificial Intelligence?"
}
```

---

## 🧠 How It Works

1. PDF is uploaded
2. Text is extracted
3. Text is split into chunks
4. Embeddings are generated
5. Stored in vector database
6. Query → similarity search
7. Context + question → Mistral model → Answer

---

## ⚡ Performance Notes

* Mistral 7B is heavy → slow on CPU
* Recommended:

  * GPU (AWS/Azure)
  * OR smaller models for faster response

---

## 📌 Future Improvements

* Streamlit UI
* Deployment (AWS / Azure)
* Model optimization (quantization)
* Multi-document support

---

## 🙌 Author

**Garlapati Venkata Sai Sathvik**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
