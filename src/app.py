import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re
from streamlit_ace import st_ace

# — Utility to clean fenced SQL and trim blank lines —
def clean_sql(raw_sql: str) -> str:
    # remove ```sql fences
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    # split into lines, drop trailing empty lines
    lines = sql.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

# ensure a writable cache directory
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# import pipeline components
from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# — Streamlit setup —
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
        uploaded = st.file_uploader("Upload .sqlite/.db", type=["sqlite", "db"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tmp.write(uploaded.read()); tmp.close()
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

# if not connected, stop here
if not schema:
    st.info("Use the sidebar to upload or connect to a database.")
    st.stop()

# show current schema
with st.expander("Current Schema", expanded=True):
    st.graphviz_chart(render_er_diagram(schema))

# — Natural language input & SQL generation —
question = st.text_input("Ask in plain English")
mode = st.selectbox("Generation mode", ["LangChain RAG", "Manual FAISS", "Schema Only"])

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

# — Editable SQL editor with padded blank lines for consistent gutter —
if "sql" in st.session_state:
    sql_text = st.session_state["sql"]
    lines = sql_text.splitlines()
    line_count = len(lines)
    extra_blank = 2
    display_lines = line_count + extra_blank
    # ~30px per line, clamp between 120px and 500px
    height_px = min(max(display_lines * 30, 120), 500)

    with st.form("sql_form"):
        st.subheader("Generated SQL (editable)")
        # pad with actual blank lines so gutter numbers appear
        padded_sql = sql_text + "\n" * extra_blank
        edited_sql = st_ace(
            value=padded_sql,
            language="sql",
            theme="github",
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            height=height_px,
            key="sql_editor"
        )
        run = st.form_submit_button("Run")

    # execute query on form submit
    if run:
        try:
            if edited_sql.strip().lower().startswith("select"):
                df = connector(edited_sql)
                st.session_state["df"] = df
            else:
                executor(edited_sql)
                st.session_state["df"] = None
                # refresh schema after write
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
