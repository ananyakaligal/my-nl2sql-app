import os
import re
import tempfile

import pandas as pd
import sqlite3
import streamlit as st
from sqlalchemy import create_engine
from streamlit_ace import st_ace

# — Gemini TextServiceClient for translation —
from google.ai import generativelanguage as glm
glm_client = glm.TextServiceClient()
glm.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def translate_to_english(text: str) -> str:
    """Use Gemini (chat-bison-001) to translate into English."""
    prompt = glm.TextPrompt(
        text=f"Translate this to clear, idiomatic English:\n\n{text}"
    )
    response = glm_client.generate_text(
        model="chat-bison-001",
        prompt=prompt,
    )
    # `output` holds the translated text
    return response.candidates[0].output.strip()

# — Clean SQL fences & trailing blanks —
def clean_sql(raw_sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    lines = sql.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

# — Ensure pip cache dir exists in HF Spaces —
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# — Import your existing NL→SQL pipeline pieces —
from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import (
    generate_sql_from_prompt,
    generate_sql_schema_only,
)
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# — Streamlit page setup —
st.set_page_config(page_title="Text-to-SQL", layout="wide")
st.title("Text-to-SQL")

# — Sidebar: Database connection —
with st.sidebar:
    st.header("1) Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema = connector = executor = conn = None
    db_path = uri = None

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
        with st.expander("Connection Info"):
            host = st.text_input("Host")
            port = st.text_input(
                "Port", "5432" if db_type == "PostgreSQL" else "3306"
            )
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
                engine = create_engine(uri)
                conn = engine.connect()
                schema = extract_schema_rdbms(uri)
                connector = lambda q: pd.read_sql_query(q, conn)
                executor = lambda q: conn.execute(q)
                st.success("Connected")
            except Exception as e:
                st.error(e)

# — Stop if no DB connected —
if not schema:
    st.info("Use the sidebar to connect or upload your database.")
    st.stop()

# — Show current schema —
with st.expander("Current Schema", expanded=True):
    st.graphviz_chart(render_er_diagram(schema))

# — Step 2: Enter question & choose language —
st.subheader("2) Enter Your Question")
lang = st.selectbox(
    "Input Language",
    ["English", "Spanish", "French", "German", "Chinese", "Hindi", "Japanese", "Other"]
)
raw_q = st.text_area("Question", height=80)

# — Step 3: Generate SQL (translating if needed) —
mode = st.selectbox(
    "Generation Mode", ["LangChain RAG", "Manual FAISS", "Schema Only"]
)
if st.button("Generate SQL"):
    # translate only if not English
    prompt = (
        translate_to_english(raw_q)
        if lang != "English" and raw_q.strip()
        else raw_q
    ).strip()

    # generate raw SQL
    if mode == "LangChain RAG":
        raw_sql = generate_sql_with_langchain(prompt, schema)
    elif mode == "Manual FAISS":
        idx, meta = build_or_load_index(schema)
        raw_sql = generate_sql_from_prompt(prompt, idx, meta, schema)
    else:
        raw_sql = generate_sql_schema_only(prompt, schema)

    st.session_state["sql"] = clean_sql(raw_sql)
    st.session_state["ran"] = False

# — Step 4: Editable SQL & Run —
if "sql" in st.session_state:
    sql_text = st.session_state["sql"]
    lines = sql_text.splitlines()
    extra_blank = 2
    min_lines = len(lines) + extra_blank

    with st.form("sql_form"):
        st.subheader("3) Review / Edit Generated SQL")
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
                schema = (
                    extract_schema_sqlite(db_path)
                    if db_type == "SQLite"
                    else extract_schema_rdbms(uri)
                )
                st.session_state["schema"] = schema
            st.session_state["ran"] = True
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.session_state["ran"] = False

# — Step 5: Display Results or Updated Schema —
if st.session_state.get("ran", False):
    if st.session_state.get("df") is not None:
        st.subheader("4) Query Results")
        st.dataframe(st.session_state["df"], use_container_width=True)
    else:
        st.subheader("4) Schema Updated")
        st.graphviz_chart(render_er_diagram(st.session_state["schema"]))
        with st.expander("Table Previews"):
            for tbl in st.session_state["schema"]:
                st.write(f"**{tbl}**")
                preview = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
                st.dataframe(preview, use_container_width=True)
