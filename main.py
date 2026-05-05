from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import os
import torch

from dotenv import load_dotenv
load_dotenv()

from backend.services.pdf_loader import load_pdf
from backend.services.chunker import chunk_text
from backend.services.embeddings import get_embeddings
from backend.services.vector_store import VectorStore

from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()

# -----------------------------
# CPU OPTIMIZATION
# -----------------------------
device = "cpu"
torch.set_num_threads(os.cpu_count())

print("⚡ Optimized Mistral CPU mode")
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
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).to(device)

model.eval()

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "⚡ Fast Mistral chatbot running"}

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
        if not text:
            return {"error": "No text extracted from PDF"}

        chunks = chunk_text(text)
        if not chunks:
            return {"error": "Chunking failed"}

        embeddings = get_embeddings(chunks)
        vector_store.add(embeddings, chunks)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "message": "PDF uploaded successfully"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------------
# CHAT API (IMPROVED)
# -----------------------------
@app.post("/chat")
def chat(query: QueryRequest):
    try:
        # 1. Retrieve context (fast)
        query_embedding = get_embeddings([query.question])[0]
        chunks = vector_store.search(query_embedding, k=1)

        context = chunks[0] if chunks else "No relevant context available"

        # 2. Better prompt (less echo)
        prompt = f"Answer clearly using context.\n\nContext: {context}\n\nQ: {query.question}\nA:"

        # 3. Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 4. Generate (FIXED)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,   # 🔥 FIX: prevent cut answers
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🔥 STRONG CLEANING
        if "A:" in full_output:
            answer = full_output.split("A:")[-1]
        else:
            answer = full_output

        # remove noise
        for bad in ["Context:", "Q:", "Question:"]:
            if bad in answer:
                answer = answer.split(bad)[0]

        # clean + first line only
        answer = answer.strip().split("\n")[0]

        # ensure proper ending
        if not answer.endswith("."):
            answer += "."

        return {
            "question": query.question,
            "answer": answer,
            "relevant_chunks": chunks
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}