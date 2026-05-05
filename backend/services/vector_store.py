import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        # Ensure correct numpy format
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding]).astype("float32")

        D, I = self.index.search(query_embedding, k)

        results = []
        seen = set()

        for idx in I[0]:
            if idx == -1:
                continue
            if idx not in seen:
                results.append(self.texts[idx])
                seen.add(idx)

        return results