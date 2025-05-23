import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ─── Hugging Face cache & (optional) login ──────────────────────────────────────
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["TRANSFORMERS_CACHE"],
)

# ─── Default vectorstore dir (if no override) ──────────────────────────────────
_default_vs_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../vectorstore")
)


def build_or_load_index(schema_dict):
    """
    Load a FAISS index + metadata if they exist, else build, save, and return them.
    Honors VECTORSTORE_DIR_OVERRIDE for testing.
    """
    # pick up override at call-time
    vs_dir = os.getenv("VECTORSTORE_DIR_OVERRIDE", _default_vs_dir)
    os.makedirs(vs_dir, exist_ok=True)

    idx_path = os.path.join(vs_dir, "schema_index.faiss")
    meta_path = os.path.join(vs_dir, "schema_meta.pkl")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        print("🔄 Loading FAISS index and metadata from disk...")
        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    print("⚡ Building new FAISS index and metadata (cold start)...")
    texts, metadata = [], []
    for table, cols in schema_dict.items():
        for col in cols:
            texts.append(f"{table} - {col}")
            metadata.append({"table": table, "column": col})

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata
