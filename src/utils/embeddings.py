import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Ensure correct Streamlit and cache directories
os.environ["STREAMLIT_HOME"] = os.getenv("STREAMLIT_HOME", "/tmp/.streamlit")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache")

# Login to Hugging Face using token (only if provided)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Load model with specified cache folder
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["TRANSFORMERS_CACHE"]
)

# Set up FAISS index paths
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

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata
