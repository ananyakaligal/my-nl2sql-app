# langchain_sql_pipeline.py

import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from gemini_flash_beta_llm import GeminiFlashBetaLLM

# Load environment variables (for GEMINI_API_KEY, etc.)
load_dotenv()

# Compute project root (one level above src/)
ROOT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
META_PATH = VECTORSTORE_DIR / "schema_meta.pkl"

def load_faiss_retriever():
    """
    Load FAISS index with HuggingFace embeddings for column-level retrieval.
    """
    if not META_PATH.exists():
        raise FileNotFoundError(f"FAISS metadata not found at {META_PATH}")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    texts = [f"{m['table']} - {m['column']}" for m in metadata]
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embedder)
    return db.as_retriever(search_type="similarity", k=5), metadata

def generate_sql_with_langchain(user_query: str, schema_dict: dict) -> str:
    """
    RAG pipeline: uses FAISS for retrieval of relevant columns,
    then Gemini-Flash (v1beta) to generate a SQL query.
    """
    # 1) Retrieve relevant columns
    retriever, metadata = load_faiss_retriever()
    docs = retriever.get_relevant_documents(user_query)
    semantic_context = "\n".join(d.page_content for d in docs)

    # 2) Format full schema for grounding
    schema_text = "\n".join(
        f"Table: {t} — Columns: {', '.join(cols)}"
        for t, cols in schema_dict.items()
    )

    # 3) Build prompt template
    prompt_template = PromptTemplate.from_template("""
You are a SQL expert. Given the database schema and relevant columns, write a SQL query for the user question.

### DATABASE SCHEMA
{schema}

### RELEVANT COLUMNS
{context}

### USER QUESTION
{question}

Only use valid table and column names. Do not hallucinate.

SQL:
""")

    # 4) Instantiate the Flash-beta LLM wrapper
    llm = GeminiFlashBetaLLM()

    # 5) Create and run the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(schema=schema_text, context=semantic_context, question=user_query)

# ----------------------------------------
# Optional CLI test
# ----------------------------------------
if __name__ == "__main__":
    example_schema = {
        "users": ["id", "name", "age"],
        "orders": ["order_id", "user_id", "amount", "created_at"]
    }
    question = "Find all users older than 30 who placed orders over 100"
    print(generate_sql_with_langchain(question, example_schema))
