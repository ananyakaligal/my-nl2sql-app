import sqlite3
import pandas as pd
import tempfile

def test_connector_and_executor_work_together():
    # 1. Create an on-disk SQLite DB
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute("CREATE TABLE t(val INTEGER);")
    conn.executemany("INSERT INTO t(val) VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()

    # 2. Define connector + executor
    connector = lambda q: pd.read_sql_query(q, conn)
    def executor(q):
        c = conn.cursor(); c.execute(q); conn.commit(); return c

    # 3. Test SELECT via connector
    df = connector("SELECT val FROM t ORDER BY val;")
    assert df["val"].tolist() == [1, 2, 3]

    # 4. Test INSERT via executor
    executor("INSERT INTO t(val) VALUES (4);")
    df2 = connector("SELECT COUNT(*) AS cnt FROM t;")
    assert int(df2["cnt"][0]) == 4

    conn.close()
