import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re

# --- Ensure vectorstore directory exists relative to src/app.py ---
vectorstore_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectorstore"))
os.makedirs(vectorstore_dir, exist_ok=True)

# If you need the FAISS index path, use:
index_path = os.path.join(vectorstore_dir, "schema_index.faiss")

from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import (
    generate_sql_from_prompt,
    generate_sql_schema_only,
)
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# --- SQL cleaning utility ---
def clean_sql(raw_sql):
    """
    Removes markdown code block markers from the generated SQL.
    Handles ```sql ... ```
    """
    # Remove ```sql or ``` from start
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    # Remove trailing ```
    sql = re.sub(r"```$", "", sql)
    return sql.strip()

# --- Page setup ---
st.set_page_config(
    page_title="Text-to-SQL RAG Demo",
    layout="wide",
)

# Hide Streamlit chrome
st.markdown(
    """
    <style>
      #MainMenu, header, footer { visibility: hidden; }
      .css-12oz5g7, .block-container { background: #000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar for DB setup ---
with st.sidebar:
    st.header("Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema_data = None
    db_connector = None

    if db_type == "SQLite":
        uploaded = st.file_uploader("Upload .db/.sqlite/.sql", type=["db", "sqlite", "sql"])
        if uploaded:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tf.write(uploaded.read())
            tf.close()
            schema_data = extract_schema_sqlite(tf.name)
            db_connector = lambda q: pd.read_sql_query(q, sqlite3.connect(tf.name))

    else:
        with st.expander("Enter credentials"):
            host = st.text_input("Host", key="host")
            port = st.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306", key="port")
            user = st.text_input("User", key="user")
            pwd = st.text_input("Password", type="password", key="pwd")
            name = st.text_input("Database", key="dbname")
        if st.button("Connect", key="connect_btn"):
            uri = (
                f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
                if db_type == "PostgreSQL"
                else f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}"
            )
            try:
                schema_data = extract_schema_rdbms(uri)
                engine = create_engine(uri)
                db_connector = lambda q: pd.read_sql_query(q, engine)
                st.success("Connected")
            except Exception as e:
                st.error(f"{e}")

# --- Main panel ---
st.title("Text-to-SQL Generator")

if schema_data:
    # Schema diagram
    st.subheader("Schema Diagram")
    st.graphviz_chart(render_er_diagram(schema_data))

    # Question + mode
    st.subheader("Ask Your Database")
    q_col, m_col = st.columns((3, 1))
    with q_col:
        question = st.text_input(
            "Your Question",
            placeholder="e.g. List rock-genre tracks",
            key="user_question",
            label_visibility="collapsed"
        )
    with m_col:
        mode = st.selectbox(
            "Mode",
            ["LangChain RAG", "Manual FAISS", "Schema Only"],
            help="How to generate the SQL",
            key="mode_select"
        )

    # Generate button
    generate = st.button("Generate SQL", use_container_width=True, key="generate_btn")

    if generate and question:
        with st.spinner("Generating…"):
            if mode == "LangChain RAG":
                raw = generate_sql_with_langchain(question, schema_data)
            elif mode == "Manual FAISS":
                idx, meta = build_or_load_index(schema_data)
                raw = generate_sql_from_prompt(question, idx, meta, schema_data)
            else:
                raw = generate_sql_schema_only(question, schema_data)
        sql = clean_sql(raw)

        # Show SQL
        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        # Show results
        if db_connector:
            st.subheader("Results")
            try:
                df = db_connector(sql)
                st.metric("Rows returned", len(df))
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Execution failed: {e}")

else:
    st.info("Use the sidebar to upload or connect to a database.")
