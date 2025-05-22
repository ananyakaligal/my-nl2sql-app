import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re
from streamlit_ace import st_ace

# Ensure writable cache dirs
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# Vectorstore path
vectorstore_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectorstore"))
os.makedirs(vectorstore_dir, exist_ok=True)

index_path = os.path.join(vectorstore_dir, "schema_index.faiss")

from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

def clean_sql(raw_sql):
    sql = re.sub(r"^```(?:sql)?\\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    return sql.strip()

st.set_page_config(page_title="Text-to-SQL RAG Demo", layout="wide")

st.markdown("""
<style>
  #MainMenu, header, footer { visibility: hidden; }
  .css-12oz5g7, .block-container { background: #000 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema_data = None
    db_connector = None
    db_path = None

    if db_type == "SQLite":
        uploaded = st.file_uploader("Upload .db/.sqlite/.sql", type=["db", "sqlite", "sql"])
        if uploaded:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tf.write(uploaded.read())
            tf.close()
            db_path = tf.name
            schema_data = extract_schema_sqlite(db_path)
            db_connector = lambda q: pd.read_sql_query(q, sqlite3.connect(db_path))
            db_executor = lambda q: sqlite3.connect(db_path).cursor().execute(q)
    else:
        with st.expander("Enter credentials"):
            host = st.text_input("Host")
            port = st.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306")
            user = st.text_input("User")
            pwd = st.text_input("Password", type="password")
            name = st.text_input("Database")
        if st.button("Connect"):
            uri = (
                f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
                if db_type == "PostgreSQL"
                else f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}"
            )
            try:
                schema_data = extract_schema_rdbms(uri)
                engine = create_engine(uri)
                db_connector = lambda q: pd.read_sql_query(q, engine)
                db_executor = lambda q: engine.execute(q)
                st.success("Connected")
            except Exception as e:
                st.error(f"{e}")

st.title("Text-to-SQL Generator")

if schema_data:
    st.markdown(f"**Connected Database:** `{db_type}`")

    st.subheader("Schema Diagram")
    st.graphviz_chart(render_er_diagram(schema_data))

    st.subheader("Ask Your Database")
    q_col, m_col = st.columns((3, 1))
    with q_col:
        question = st.text_input("Your Question", placeholder="e.g. List rock-genre tracks", label_visibility="collapsed")
    with m_col:
        mode = st.selectbox("Mode", ["LangChain RAG", "Manual FAISS", "Schema Only"])

    if st.button("Generate SQL"):
        with st.spinner("Generating…"):
            if mode == "LangChain RAG":
                raw = generate_sql_with_langchain(question, schema_data)
            elif mode == "Manual FAISS":
                idx, meta = build_or_load_index(schema_data)
                raw = generate_sql_from_prompt(question, idx, meta, schema_data)
            else:
                raw = generate_sql_schema_only(question, schema_data)
        sql = clean_sql(raw)

        st.subheader("Generated SQL (editable)")
        edited_sql = st_ace(value=sql + "\n\n", language="sql", theme="dracula", key="sql_editor")

        if db_connector and st.button("Run Query"):
            st.subheader("Results")
            try:
                df = db_connector(edited_sql)
                st.metric("Rows returned", len(df))
                st.dataframe(df, use_container_width=True)

                # Updated schema
                if db_path:
                    schema_data = extract_schema_sqlite(db_path)
                    st.subheader("Updated Schema Diagram")
                    st.graphviz_chart(render_er_diagram(schema_data))
            except Exception as e:
                st.error(f"Execution failed: {e}")
else:
    st.info("Use the sidebar to upload or connect to a database.")
