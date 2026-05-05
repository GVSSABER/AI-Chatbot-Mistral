from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import os
import torch

from dotenv import load_dotenv
load_dotenv()

from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import get_embeddings
from services.vector_store import VectorStore

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()

# -----------------------------
# CPU OPTIMIZATION
# -----------------------------
device = "cpu"
torch.set_num_threads(os.cpu_count())

print("⚡ FLAN-T5 Accuracy Boosted Mode")
print("CPU Cores:", os.cpu_count())

# -----------------------------
# STORAGE
# -----------------------------
UPLOAD_FOLDER = "data/pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = VectorStore(dim=384)

# -----------------------------
# MODEL SETUP
# -----------------------------
model_id = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

model.eval()

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "⚡ AI Chatbot (Improved Accuracy Mode)"}

# -----------------------------
# UPLOAD PDF
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = load_pdf(file_path)
        chunks = chunk_text(text)

        embeddings = get_embeddings(chunks)
        vector_store.add(embeddings, chunks)

        return {
            "message": "PDF uploaded successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------------
# CHAT (FINAL ACCURATE VERSION)
# -----------------------------
@app.post("/chat")
def chat(query: QueryRequest):
    try:
        # 1. BETTER RETRIEVAL (IMPORTANT FIX)
        query_embedding = get_embeddings([query.question])[0]

        chunks = vector_store.search(query_embedding, k=4)

        # merge properly (NOT just join blindly)
        context = " ".join([c.strip() for c in chunks if c])

        # 2. STRONG PROMPT (prevents short answers)
        prompt = f"""
You are a highly intelligent AI assistant.

Read the context carefully and answer clearly.

Do NOT copy only one sentence.
Explain the full idea in simple language.

Context:
{context}

Question:
{query.question}

Answer in 4-5 detailed sentences:
"""

        # 3. Tokenization
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 4. GENERATION (QUALITY BOOSTED)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=8,           # stronger reasoning
                length_penalty=1.4,    # encourages long answers
                early_stopping=True
            )

        # 5. Decode safely
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 6. CLEAN OUTPUT (safe cleanup only)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        if "Question:" in answer:
            answer = answer.split("Question:")[-1].strip()

        return {
            "question": query.question,
            "answer": answer,
            "relevant_chunks": chunks
        }

    except Exception as e:
        return {"error": str(e)}
