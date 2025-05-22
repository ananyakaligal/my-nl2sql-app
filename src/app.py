import os
import re
import tempfile

import pandas as pd
import sqlite3
import streamlit as st
from sqlalchemy import create_engine
from streamlit_ace import st_ace

import google.generativeai as genai

# — Configure Gemini (Google Generative AI) —
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# — Utility to clean fenced SQL and trim blank lines —
def clean_sql(raw_sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    lines = sql.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

# — Translate any language into English via Gemini —
def translate_to_english(text: str) -> str:
    resp = genai.chat.completions.create(
        model="models/chat-bison-001",
        prompt_messages=[{"author": "user", "content": f"Translate this to clear English:\n\n{text}"}]
    )
    return resp.completions[0].content.strip()

# ensure cache dir writable
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# import NL2SQL pipeline components
from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# — Streamlit page setup —
st.set_page_config(page_title="Text-to-SQL", layout="wide")
st.title("Text-to-SQL")

# — Sidebar: connect to database —
with st.sidebar:
    st.header("Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema = connector = executor = conn = None
    db_path = None
    uri = None

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
                c = conn.cursor(); c.execute(q); conn.commit(); return c
            executor = _exec

    else:
        with st.expander("Connection info"):
            host = st.text_input("Host")
            port = st.text_input("Port", "5432" if db_type=="PostgreSQL" else "3306")
            user = st.text_input("User")
            pwd  = st.text_input("Password", type="password")
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

# stop if no connection
if not schema:
    st.info("Use the sidebar to connect or upload your database.")
    st.stop()

# show current schema
with st.expander("Current Schema", expanded=True):
    st.graphviz_chart(render_er_diagram(schema))

# — Step 1: Enter question & pick language —
st.subheader("Enter Your Question")
lang = st.selectbox(
    "Input Language",
    ["English", "Spanish", "French", "German", "Chinese", "Hindi", "Japanese", "Other"]
)
raw_q = st.text_area("Question", height=80)

# — Step 2: Generate SQL (with on-the-fly translation) —
mode = st.selectbox("Generation Mode", ["LangChain RAG", "Manual FAISS", "Schema Only"])
if st.button("Generate SQL"):
    # decide whether to translate
    if lang != "English" and raw_q.strip():
        prompt = translate_to_english(raw_q)
    else:
        prompt = raw_q

    st.session_state["sql"] = clean_sql(
        generate_sql_with_langchain(prompt, schema)
        if mode=="LangChain RAG" else
        (generate_sql_from_prompt(*build_or_load_index(schema), schema) if mode=="Manual FAISS" 
          else generate_sql_schema_only(prompt, schema))
    )
    st.session_state["ran"] = False

# — Step 3: Editable SQL & Run —
if "sql" in st.session_state:
    sql_text = st.session_state["sql"]
    lines = sql_text.splitlines()
    extra_blank = 2
    min_lines = len(lines) + extra_blank

    with st.form("sql_form"):
        st.subheader("Generated SQL (editable)")
        edited = st_ace(
            value=sql_text,
            language="sql",
            theme="github",
            show_gutter=True,
            wrap=True,
            min_lines=min_lines,
            key="sql_editor"
        )
        run = st.form_submit_button("Run Query")

    if run:
        try:
            if edited.strip().lower().startswith("select"):
                df = connector(edited)
                st.session_state["df"] = df
            else:
                executor(edited)
                st.session_state["df"] = None
                # refresh schema
                schema = extract_schema_sqlite(db_path) if db_type=="SQLite" else extract_schema_rdbms(uri)
                st.session_state["schema"] = schema
            st.session_state["ran"] = True
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.session_state["ran"] = False

# — Step 4: Display results or updated schema —
if st.session_state.get("ran", False):
    if st.session_state.get("df") is not None:
        st.subheader("Results")
        st.dataframe(st.session_state["df"], use_container_width=True)
    else:
        st.subheader("Schema Updated")
        st.graphviz_chart(render_er_diagram(st.session_state["schema"]))
        with st.expander("Table Previews"):
            for tbl in st.session_state["schema"]:
                st.write(f"**{tbl}**")
                preview = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
                st.dataframe(preview, use_container_width=True)
