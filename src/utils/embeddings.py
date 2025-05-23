import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ─── Hugging Face cache & login ────────────────────────────────────────────────
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["TRANSFORMERS_CACHE"],
)

# ─── Vectorstore directory override for tests ──────────────────────────────────
OVERRIDE = os.getenv("VECTORSTORE_DIR_OVERRIDE")
if OVERRIDE:
    vectorstore_dir = OVERRIDE
else:
    vectorstore_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../vectorstore")
    )
os.makedirs(vectorstore_dir, exist_ok=True)

# ─── FAISS index & metadata paths ──────────────────────────────────────────────
index_path = os.path.join(vectorstore_dir, "schema_index.faiss")
meta_path = os.path.join(vectorstore_dir, "schema_meta.pkl")


def build_or_load_index(schema_dict):
    """
    Either load an existing FAISS index + metadata pickle from disk
    or build a new one, write it out, and return it.
    """
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

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata
