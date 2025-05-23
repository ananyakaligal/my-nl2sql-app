# src/utils/embeddings.py
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ─── Hugging Face cache & optional login ────────────────────────────────────────
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["TRANSFORMERS_CACHE"],
)

# ─── Determine vectorstore directory (allow override for tests) ───────────────
_default_vs_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../vectorstore")
)
_vs_dir = os.getenv("VECTORSTORE_DIR_OVERRIDE", _default_vs_dir)
os.makedirs(_vs_dir, exist_ok=True)

# ─── Expose these so tests can import them ─────────────────────────────────────
index_path = os.path.join(_vs_dir, "schema_index.faiss")
meta_path  = os.path.join(_vs_dir, "schema_meta.pkl")


def build_or_load_index(schema_dict):
    """
    Either load an existing FAISS index + metadata from disk,
    or build a new one, write it out, and return it.
    """
    # If both files exist, load from disk
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("🔄 Loading FAISS index and metadata from disk…")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    # Otherwise, build from scratch
    print("⚡ Building new FAISS index and metadata (cold start)…")
    texts, metadata = [], []
    for table, cols in schema_dict.items():
        for col in cols:
            texts.append(f"{table} - {col}")
            metadata.append({"table": table, "column": col})

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    # Persist to disk
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata
