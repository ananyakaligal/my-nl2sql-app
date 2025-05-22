# llm_sql_generator.py

import os
import re
import sqlite3
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from gemini_flash_beta_llm import GeminiFlashBetaLLM

# ----------------------------------------
# Initialization
# ----------------------------------------
load_dotenv()

# Instantiate the Gemini Flash-beta wrapper (LangChain LLM)
llm = GeminiFlashBetaLLM()

# Embedding model for semantic search
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------
# Utility Functions
# ----------------------------------------

def clean_sql_output(llm_text: str) -> str:
    """
    Remove markdown-style SQL blocks and clean whitespace.
    """
    return re.sub(r"```sql|```", "", llm_text, flags=re.IGNORECASE).strip()


def extract_schema_from_db(db_path: str) -> dict:
    """
    Extracts table and column metadata from a SQLite DB file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [col_info[1] for col_info in cursor.fetchall()]
        schema[table] = cols
    conn.close()
    return schema


def format_schema_for_prompt(schema: dict) -> str:
    """
    Formats schema into a prompt-friendly string.
    """
    return "\n".join([
        f"Table: {table} — Columns: {', '.join(columns)}"
        for table, columns in schema.items()
    ])

@lru_cache(maxsize=1000)
def get_query_embedding(text: str):
    """
    Cacheable embedding for semantic search via SentenceTransformer.
    """
    return embed_model.encode([text])[0]

# ----------------------------------------
# SQL Generation Functions
# ----------------------------------------

def generate_sql_from_prompt(
    user_query: str,
    index,
    metadata,
    schema_dict: dict,
    top_k: int = 5
) -> str:
    """
    Generate SQL using Gemini Flash-beta + semantic retrieval + schema.
    """
    # 1) Semantic retrieval
    query_emb = [get_query_embedding(user_query)]
    D, I = index.search(np.array(query_emb).astype("float32"), top_k)

    semantic_context = []
    for idx in I[0]:
        item = metadata[idx]
        semantic_context.append(f"Table: {item['table']} | Column: {item['column']}")
    semantic_text = "\n".join(semantic_context)

    # 2) Schema context
    schema_text = format_schema_for_prompt(schema_dict)

    # 3) Build prompt
    prompt = f"""
You are a SQL expert.

Use only the following schema and column names when generating SQL.

### DATABASE SCHEMA
{schema_text}

### RELEVANT COLUMNS
{semantic_text}

### USER QUESTION
{user_query}

Generate valid SQL using only the exact schema and column names provided.
Do not hallucinate table or column names.

SQL:
"""

    # 4) Call the LLM wrapper
    try:
        response = llm._call(prompt)
        return clean_sql_output(response)
    except Exception as e:
        print(f"[Flash LLM Error] {e}")
        return "-- LLM failed to respond --"


def generate_sql_schema_only(
    user_query: str,
    schema_dict: dict
) -> str:
    """
    Fallback generation using only full schema, no retrieval.
    """
    schema_text = format_schema_for_prompt(schema_dict)
    prompt = f"""
You are a SQL expert.

Below is the full schema of the database:

{schema_text}

User question:
{user_query}

Write a valid SQL query using only the tables and columns from the schema above.
Do not hallucinate table or column names.

SQL:
"""

    try:
        response = llm._call(prompt)
        return clean_sql_output(response)
    except Exception as e:
        print(f"[Flash LLM Error] {e}")
        return "-- LLM failed to respond --"

# ----------------------------------------
# Local Test CLI
# ----------------------------------------
if __name__ == "__main__":
    # Example SQLite schema extraction
    schema = extract_schema_from_db("your.db")
    question = "List all customers with purchases after 2023"
    print(generate_sql_schema_only(question, schema))