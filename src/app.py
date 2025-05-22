import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re
from streamlit_ace import st_ace

# === Ensure writable cache dirs ===
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)

# --- Vectorstore setup ---
vectorstore_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../vectorstore")
)
os.makedirs(vectorstore_dir, exist_ok=True)
index_path = os.path.join(vectorstore_dir, "schema_index.faiss")

from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import (
    generate_sql_from_prompt,
    generate_sql_schema_only,
)
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

def clean_sql(raw_sql: str) -> str:
    """Strip any ```sql fences."""
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    return sql.strip()

st.set_page_config(page_title="Text-to-SQL RAG Demo", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu, header, footer { visibility: hidden; }
      .css-12oz5g7, .block-container { background: #000 !important; }
    </style>
    """, unsafe_allow_html=True
)

# --- Sidebar: Database Setup ---
with st.sidebar:
    st.header("Database Setup")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema_data = None
    db_connector = None
    db_executor = None
    db_path = None
    conn = None

    if db_type == "SQLite":
        uploaded = st.file_uploader(
            "Upload .db/.sqlite/.sql", type=["db", "sqlite", "sql"]
        )
        if uploaded:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tf.write(uploaded.read())
            tf.close()
            db_path = tf.name
            conn = sqlite3.connect(db_path)
            schema_data = extract_schema_sqlite(db_path)
            db_connector = lambda q: pd.read_sql_query(q, conn)
            def _exec(q):
                cur = conn.cursor()
                cur.execute(q)
                conn.commit()
                return cur
            db_executor = _exec

    else:
        with st.expander("Enter credentials"):
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
                schema_data = extract_schema_rdbms(uri)
                db_connector = lambda q: pd.read_sql_query(q, conn)
                def _exec(q):
                    conn.execute(q)
                    return conn
                db_executor = _exec
                st.success("Connected")
            except Exception as e:
                st.error(f"{e}")

# --- Main Interface ---
st.title("Text-to-SQL Generator")

if schema_data:
    st.markdown(f"**Connected Database:** `{db_type}`")
    st.subheader("Schema Diagram")
    st.graphviz_chart(render_er_diagram(schema_data))

    st.subheader("Ask Your Database")
    question = st.text_input(
        "Your Question",
        placeholder="e.g. List rock-genre tracks",
        key="user_question",
        label_visibility="collapsed"
    )
    mode = st.selectbox(
        "Mode", ["LangChain RAG", "Manual FAISS", "Schema Only"], key="mode_select"
    )

    # 1) Generate SQL and store in session_state
    if st.button("Generate SQL", key="gen_sql"):
        with st.spinner("Generating…"):
            if mode == "LangChain RAG":
                raw = generate_sql_with_langchain(question, schema_data)
            elif mode == "Manual FAISS":
                idx, meta = build_or_load_index(schema_data)
                raw = generate_sql_from_prompt(question, idx, meta, schema_data)
            else:
                raw = generate_sql_schema_only(question, schema_data)
        st.session_state["last_sql"] = clean_sql(raw)
        st.session_state["ran"] = False  # reset run state

    # 2) Display editor + Run Query in a form
    if "last_sql" in st.session_state:
        with st.form("sql_form"):
            edited_sql = st_ace(
                value=st.session_state["last_sql"],
                language="sql",
                theme="dracula",
                height=180,
                key="sql_editor",
                show_gutter=True,
                show_print_margin=False,
                wrap=True,
            )
            run = st.form_submit_button("Run Query")

        # 3) Execute on form submit
        if run:
            try:
                sql_trim = edited_sql.strip().lower()
                if sql_trim.startswith("select"):
                    df = db_connector(edited_sql)
                    st.session_state["last_df"] = df
                else:
                    db_executor(edited_sql)
                    st.session_state["last_df"] = None
                    # refresh schema
                    if db_type == "SQLite":
                        schema_data = extract_schema_sqlite(db_path)
                    else:
                        schema_data = extract_schema_rdbms(uri)
                    st.session_state["last_schema"] = schema_data
                st.session_state["ran"] = True
            except Exception as e:
                st.error(f"Execution failed: {e}")
                st.session_state["ran"] = False

    # 4) Show results or updated schema
    if st.session_state.get("ran", False):
        if st.session_state.get("last_df") is not None:
            st.subheader("Results")
            st.metric("Rows returned", len(st.session_state["last_df"]))
            st.dataframe(st.session_state["last_df"], use_container_width=True)
        else:
            st.subheader("Query executed successfully!")
            st.subheader("Updated Schema Diagram")
            st.graphviz_chart(render_er_diagram(st.session_state["last_schema"]))
            st.subheader("Table Previews (up to 5 rows)")
            for table, _ in st.session_state["last_schema"].items():
                st.markdown(f"**{table}**")
                preview = pd.read_sql_query(
                    f"SELECT * FROM {table} LIMIT 5",
                    conn
                )
                st.dataframe(preview, use_container_width=True)
else:
    st.info("Use the sidebar to upload or connect to a database.")
