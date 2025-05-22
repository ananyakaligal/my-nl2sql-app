import os
import re
import tempfile

import pandas as pd
import sqlite3
import streamlit as st
from sqlalchemy import create_engine
from streamlit_ace import st_ace

# ─── Your existing Gemini wrapper ───────────────────────────────────────────────
from gemini_flash_beta_llm import GeminiFlashBetaLLM
translator = GeminiFlashBetaLLM()

def translate_to_english(text: str) -> str:
    """Translate any-language text into English via Gemini Flash."""
    prompt = f"Translate the following text into English:\n\n{text}"
    return translator(prompt).strip()

def clean_sql(raw_sql: str) -> str:
    """Strip markdown fences and drop trailing blank lines."""
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    lines = sql.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

# ─── Ensure Hugging Face cache dir exists ───────────────────────────────────────
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# ─── Vectorstore setup ──────────────────────────────────────────────────────────
vectorstore_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectorstore"))
os.makedirs(vectorstore_dir, exist_ok=True)
index_path = os.path.join(vectorstore_dir, "schema_index.faiss")

from utils.schema_extractor    import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings          import build_or_load_index
from utils.llm_sql_generator   import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline    import generate_sql_with_langchain
from utils.er_diagram          import render_er_diagram

# ─── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Text-to-SQL RAG Demo", layout="wide")
st.title("Text-to-SQL Generator")

#
# ─── Sidebar: DATABASE CONNECTION ────────────────────────────────────────────────
#
with st.sidebar:
    st.header("1) Database Setup")
    db_type      = st.selectbox("DB Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema_data  = connector = executor = conn = None
    db_path      = uri = None

    if db_type == "SQLite":
        uploaded = st.file_uploader("Upload .sqlite/.db", type=["sqlite", "db"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tmp.write(uploaded.read()); tmp.close()
            db_path     = tmp.name
            conn        = sqlite3.connect(db_path)
            schema_data = extract_schema_sqlite(db_path)
            connector   = lambda q: pd.read_sql_query(q, conn)
            def _exec(q):
                c = conn.cursor(); c.execute(q); conn.commit(); return c
            executor = _exec

    else:
        with st.expander("Connection Info"):
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
                engine      = create_engine(uri)
                conn        = engine.connect()
                schema_data = extract_schema_rdbms(uri)
                connector   = lambda q: pd.read_sql_query(q, conn)
                executor    = lambda q: conn.execute(q)
                st.success("Connected")
            except Exception as e:
                st.error(e)

if not schema_data:
    st.info("Use the sidebar to connect or upload a database.")
    st.stop()

#
# ─── Show CURRENT SCHEMA ────────────────────────────────────────────────────────
#
with st.expander("Current Schema", expanded=True):
    st.graphviz_chart(render_er_diagram(schema_data))

#
# ─── Step 2: ENTER YOUR QUESTION ────────────────────────────────────────────────
#
st.subheader("2) Enter Your Question")
input_mode = st.radio("Input Mode", ["English", "Other Language"], horizontal=True)

if input_mode == "English":
    question = st.text_input(
        "Question (in English)",
        placeholder="e.g. List rock-genre tracks",
        key="user_question"
    )
else:
    lang_choice = st.selectbox(
        "Language",
        ["Spanish", "French", "German", "Chinese", "Hindi", "Japanese", "Other"],
        key="lang_choice"
    )
    question = st.text_area(
        f"Question (in {lang_choice})",
        placeholder=f"Enter your question in {lang_choice}",
        key="user_question"
    )
    if question:
        with st.spinner("Translating to English…"):
            translated = translate_to_english(question)
        st.text_area(
            "➜ English Translation",
            value=translated,
            height=80,
            disabled=True
        )

#
# ─── Step 3: GENERATE SQL ───────────────────────────────────────────────────────
#
mode = st.selectbox("3) Generation Mode", ["LangChain RAG", "Manual FAISS", "Schema Only"], key="mode_select")

if st.button("Generate SQL", key="generate_btn") and question:
    with st.spinner("Generating SQL…"):
        if mode == "LangChain RAG":
            raw = generate_sql_with_langchain(question, schema_data)
        elif mode == "Manual FAISS":
            idx, meta = build_or_load_index(schema_data)
            raw = generate_sql_from_prompt(question, idx, meta, schema_data)
        else:
            raw = generate_sql_schema_only(question, schema_data)

    st.session_state["generated_sql"] = clean_sql(raw)
    st.session_state.pop("query_df", None)
    st.session_state.pop("schema_data_updated", None)

#
# ─── Step 4: REVIEW / EDIT & RUN ────────────────────────────────────────────────
#
if "generated_sql" in st.session_state:
    st.subheader("4) Review / Edit Generated SQL")
    sql_text   = st.session_state["generated_sql"]
    lines      = sql_text.splitlines()
    edited_sql = st_ace(
        value=sql_text,
        language="sql",
        theme="github",
        show_gutter=True,
        wrap=True,
        min_lines=len(lines) + 2,
        key="sql_editor"
    )

    if st.button("Run Query", key="run_query") and connector:
        try:
            if edited_sql.strip().lower().startswith("select"):
                df = connector(edited_sql)
                st.session_state["query_df"] = df
            else:
                executor(edited_sql)
                st.session_state["schema_data_updated"] = (
                    extract_schema_sqlite(db_path)
                    if db_type == "SQLite"
                    else extract_schema_rdbms(uri)
                )
        except Exception as e:
            st.error(f"Execution failed: {e}")

#
# ─── Step 5: DISPLAY RESULTS or UPDATED SCHEMA ─────────────────────────────────
#
if "query_df" in st.session_state:
    st.subheader("5) Query Results")
    st.dataframe(st.session_state["query_df"], use_container_width=True)

elif "schema_data_updated" in st.session_state:
    st.subheader("5) Schema Updated")
    st.graphviz_chart(render_er_diagram(st.session_state["schema_data_updated"]))
    with st.expander("Table Previews"):
        for tbl in st.session_state["schema_data_updated"]:
            st.write(f"**{tbl}**")
            preview = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
            st.dataframe(preview, use_container_width=True)
