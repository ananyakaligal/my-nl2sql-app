import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from sqlalchemy import create_engine
import os
import re
from streamlit_ace import st_ace

# === CSS STYLING ===
st.markdown(
    """
    <style>
      /* Page padding */
      .app-container .main {
        padding: 2rem 3rem;
      }
      /* Sidebar background */
      .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
      }
      /* Hide footer/header */
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Helpers & Setup ===
def clean_sql(raw_sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql)
    sql = re.sub(r"```$", "", sql)
    return sql.strip()

# ensure cache dir
os.makedirs(os.getenv("TRANSFORMERS_CACHE", "/tmp/.cache"), exist_ok=True)
    
# load pipeline utils
from utils.schema_extractor import extract_schema_sqlite, extract_schema_rdbms
from utils.embeddings import build_or_load_index
from utils.llm_sql_generator import generate_sql_from_prompt, generate_sql_schema_only
from langchain_sql_pipeline import generate_sql_with_langchain
from utils.er_diagram import render_er_diagram

# --- Sidebar: Database Setup ---
with st.sidebar:
    st.header("🔌 Connection")
    db_type = st.selectbox("Type", ["SQLite", "PostgreSQL", "MySQL"])
    schema_data = None
    db_connector = None
    db_executor = None
    db_path = None
    conn = None

    if db_type == "SQLite":
        uploaded = st.file_uploader("Upload .sqlite/.db", type=["sqlite", "db"])
        if uploaded:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
            tf.write(uploaded.read()); tf.close()
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
                schema_data = extract_schema_rdbms(uri)
                db_connector = lambda q: pd.read_sql_query(q, conn)
                db_executor = lambda q: conn.execute(q)
                st.success("Connected")
            except Exception as e:
                st.error(e)

# --- Main Page ---
st.title("🛠️ Text-to-SQL RAG Tool")
st.markdown("Ask questions in plain English → see SQL, edit, run & explore results.")

if not schema_data:
    st.info("🗄️ Please set up your database in the sidebar.")
    st.stop()

# show schema on the side
st.subheader("📐 Current Schema")
st.graphviz_chart(render_er_diagram(schema_data))

# --- Tabs for Query vs Output ---
tab_query, tab_output = st.tabs(["🚀 Query", "📊 Output"])

with tab_query:
    st.subheader("🔍 Ask Your Database")
    question = st.text_input("Natural-language question", key="question_input")
    mode = st.selectbox("Generation mode", ["LangChain RAG","Manual FAISS","Schema Only"], key="mode_input")
    
    if st.button("Generate SQL", key="gen_btn"):
        with st.spinner("🤖 Generating…"):
            if mode=="LangChain RAG":
                raw = generate_sql_with_langchain(question, schema_data)
            elif mode=="Manual FAISS":
                idx, meta = build_or_load_index(schema_data)
                raw = generate_sql_from_prompt(question, idx, meta, schema_data)
            else:
                raw = generate_sql_schema_only(question, schema_data)
        st.session_state.last_sql = clean_sql(raw)
        st.session_state.ran = False

    if "last_sql" in st.session_state:
        st.subheader("📝 SQL Editor")
        edited_sql = st_ace(
            value=st.session_state.last_sql,
            language="sql", theme="github",
            height=200, key="ace"
        )
        if st.button("Run SQL", key="run_btn"):
            st.session_state.ran = True
            try:
                if edited_sql.lstrip().lower().startswith("select"):
                    st.session_state.last_df = db_connector(edited_sql)
                else:
                    db_executor(edited_sql)
                    st.session_state.last_df = None
                    # refresh schema
                    if db_type=="SQLite":
                        schema_data = extract_schema_sqlite(db_path)
                    else:
                        schema_data = extract_schema_rdbms(uri)
                    st.session_state.last_schema = schema_data
                st.success("✅ Query executed")
            except Exception as e:
                st.error(f"⚠️ {e}")
                st.session_state.ran = False

with tab_output:
    if not st.session_state.get("ran", False):
        st.info("Run a query in the **🚀 Query** tab to see results or updated schema.")
    else:
        if st.session_state.get("last_df") is not None:
            df = st.session_state.last_df
            # Metrics row
            c1, c2 = st.columns(2)
            c1.metric("Rows", len(df))
            c2.metric("Columns", len(df.columns))
            st.subheader("📄 Query Results")
            st.dataframe(df, use_container_width=True)
        else:
            # show updated schema
            st.subheader("🔄 Updated Schema")
            st.graphviz_chart(render_er_diagram(st.session_state.last_schema))
            st.subheader("🔖 Table Previews (5 rows)")
            for tbl in st.session_state.last_schema:
                st.markdown(f"**{tbl}**")
                preview = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
                st.dataframe(preview, use_container_width=True)
