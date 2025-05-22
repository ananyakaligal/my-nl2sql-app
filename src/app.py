import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re
from streamlit_ace import st_ace

# — Utility to clean fenced SQL —
def clean_sql(raw_sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    return sql.strip()

# Ensure a writable cache directory
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# Load your NL2SQL pipeline components
from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# — Streamlit page setup —
st.set_page_config(page_title="Text-to-SQL", layout="wide")
st.title("Text-to-SQL")

# — Sidebar: Database connection —
with st.sidebar:
    st.header("Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema = None
    connector = None
    executor = None
    db_path = None
    conn = None

    if db_type == "SQLite":
        file = st.file_uploader("Upload .sqlite/.db", type=["sqlite", "db"])
        if file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tmp.write(file.read()); tmp.close()
            db_path = tmp.name
            conn = sqlite3.connect(db_path)
            schema = extract_schema_sqlite(db_path)
            connector = lambda q: pd.read_sql_query(q, conn)
            def _exec(q):
                cur = conn.cursor()
                cur.execute(q)
                conn.commit()
                return cur
            executor = _exec

    else:
        with st.expander("Connection info"):
            host = st.text_input("Host")
            port = st.text_input("Port", "5432" if db_type=="PostgreSQL" else "3306")
            user = st.text_input("User")
            pwd = st.text_input("Password", type="password")
            name = st.text_input("Database")
        if st.button("Connect"):
            uri = (
                f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
                if db_type=="PostgreSQL"
                else f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}"
            )
            try:
                engine = create_engine(uri)
                conn = engine.connect()
                schema = extract_schema_rdbms(uri)
                connector = lambda q: pd.read_sql_query(q, conn)
                executor = lambda q: conn.execute(q)
                st.success("Connected")
            except Exception as e:
                st.error(e)

# If not connected, stop here
if not schema:
    st.info("Use the sidebar to upload or connect to a database.")
    st.stop()

# — Display current schema —
with st.expander("Current Schema", expanded=True):
    st.graphviz_chart(render_er_diagram(schema))

# — Natural-language input & SQL generation —
question = st.text_input("Ask in plain English", key="nl_input")
mode = st.selectbox("Generation mode", ["LangChain RAG", "Manual FAISS", "Schema Only"], key="mode")

if st.button("Generate SQL"):
    with st.spinner("Generating…"):
        if mode == "LangChain RAG":
            raw = generate_sql_with_langchain(question, schema)
        elif mode == "Manual FAISS":
            idx, meta = build_or_load_index(schema)
            raw = generate_sql_from_prompt(question, idx, meta, schema)
        else:
            raw = generate_sql_schema_only(question, schema)
    st.session_state["sql"] = clean_sql(raw)
    st.session_state["ran"] = False

# — Editable SQL editor inside a form to prevent full rerun —
if "sql" in st.session_state:
    # Remove trailing newlines and count lines
    sql_text = st.session_state["sql"].rstrip("\n")
    line_count = sql_text.count("\n") + 1
    # Compute height: ~30px per line, clamped between 120 and 400 px
    height_px = min(max(line_count * 30, 120), 400)

    with st.form("sql_form"):
        st.subheader("Generated SQL (editable)")
        edited_sql = st_ace(
            value=sql_text,
            language="sql",
            theme="github",
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            height=height_px,
            key="sql_editor"
        )
        run = st.form_submit_button("Run")

    # — Execute on form submit —
    if run:
        try:
            is_select = edited_sql.strip().lower().startswith("select")
            if is_select:
                df = connector(edited_sql)
                st.session_state["df"] = df
            else:
                executor(edited_sql)
                st.session_state["df"] = None
                # refresh schema
                if db_type == "SQLite":
                    schema = extract_schema_sqlite(db_path)
                else:
                    schema = extract_schema_rdbms(uri)
                st.session_state["schema"] = schema
            st.session_state["ran"] = True
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state["ran"] = False

# — Display results or updated schema —
if st.session_state.get("ran", False):
    if st.session_state.get("df") is not None:
        st.subheader("Results")
        st.dataframe(st.session_state["df"], use_container_width=True)
    else:
        st.subheader("Schema Updated")
        st.graphviz_chart(render_er_diagram(st.session_state["schema"]))
        with st.expander("Table Previews"):
            for table in st.session_state["schema"]:
                st.write(f"**{table}**")
                preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
                st.dataframe(preview, use_container_width=True)
