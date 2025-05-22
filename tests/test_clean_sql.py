import pytest
from src.app import clean_sql

@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT * FROM t;\n```", "SELECT * FROM t;"),
    ("```SELECT 1```", "SELECT 1"),
    ("```sql\nA;\n\n\n```", "A;"),
])
def test_clean_sql_strips_fences_and_blank_lines(raw, expected):
    assert clean_sql(raw) == expected
