from sentence_transformers import SentenceTransformer

# FORCE CPU ONLY (IMPORTANT)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)