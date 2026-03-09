"""Check metadata.db contents and sample rows."""
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / 'app' / 'data' / 'metadata.db'
print('DB path:', db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

for table in ('users','pdf_table','chunk_table'):
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
    except Exception as e:
        n = f'ERR: {e}'
    print(f"{table}: {n}")

print('\nSample chunk_table rows:')
try:
    cur.execute('SELECT chunk_id, pdf_id, page_no, substr(chunk_text,1,80) FROM chunk_table LIMIT 10')
    rows = cur.fetchall()
    for r in rows:
        print(r)
except Exception as e:
    print('Error reading chunk_table:', e)

conn.close()
