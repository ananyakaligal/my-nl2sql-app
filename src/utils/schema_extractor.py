from sqlalchemy import create_engine, inspect
import sqlite3

def extract_schema_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        schema[table] = [col[1] for col in cursor.fetchall()]
    conn.close()
    return schema

def extract_schema_rdbms(db_uri):
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    schema = {}
    for table in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns(table)]
        schema[table] = columns
    return schema
