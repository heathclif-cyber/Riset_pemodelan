import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path("D:/Apps-Dev/swint_tradev2/trading.db")
if not db_path.exists():
    print("Database not found!")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]
print("Tables:", tables)

for table in tables:
    print(f"\n--- Table: {table} ---")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10;", conn)
        print(df.to_string())
    except Exception as e:
        print(f"Error reading {table}: {e}")

conn.close()
