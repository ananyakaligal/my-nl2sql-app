import os
import shutil
from utils.embeddings import build_or_load_index, index_path, meta_path

def test_build_or_load_index_creates_files(tmp_path, monkeypatch):
    # Monkey-patch vectorstore_dir to a tmp folder so we don't pollute your repo
    fake_dir = tmp_path / "vectorstore"
    fake_dir.mkdir()
    monkeypatch.setenv("VECTORSTORE_DIR_OVERRIDE", str(fake_dir))

    # dummy schema
    schema = {"users": ["id", "email"], "orders": ["order_id", "amount"]}

    # call build
    idx, meta = build_or_load_index(schema)

    # check metadata length and FAISS index count
    assert len(meta) == 4
    assert idx.ntotal == 4

    # check files were written
    assert (fake_dir / "schema_index.faiss").exists()
    assert (fake_dir / "schema_meta.pkl").exists()

    # clean up
    shutil.rmtree(str(fake_dir))
