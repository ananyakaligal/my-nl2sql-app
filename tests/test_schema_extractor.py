import sqlite3
import tempfile
import os
from utils.schema_extractor import extract_schema_sqlite

def test_extract_schema_sqlite_creates_expected_structure(tmp_path):
    # 1. Create a tiny SQLite file
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE foo (id INTEGER, name TEXT);")
    conn.execute("CREATE TABLE bar (x REAL);")
    conn.commit()
    conn.close()

    # 2. Extract schema
    schema = extract_schema_sqlite(str(db_file))

    # 3. Assert tables and columns
    assert "foo" in schema
    assert set(schema["foo"]) == {"id", "name"}
    assert "bar" in schema
    assert schema["bar"] == ["x"]
