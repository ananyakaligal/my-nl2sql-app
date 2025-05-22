import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Always compute the vectorstore directory relative to this file
vectorstore_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectorstore"))
os.makedirs(vectorstore_dir, exist_ok=True)

index_path = os.path.join(vectorstore_dir, "schema_index.faiss")
meta_path = os.path.join(vectorstore_dir, "schema_meta.pkl")

def build_or_load_index(schema_dict):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("🔄 Loading FAISS index and metadata from disk...")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    print("⚡ Building new FAISS index and metadata (cold start)...")
    texts, metadata = [], []
    for table, cols in schema_dict.items():
        for col in cols:
            desc = f"{table} - {col}"
            texts.append(desc)
            metadata.append({"table": table, "column": col})

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    # Directory is already ensured above
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata
