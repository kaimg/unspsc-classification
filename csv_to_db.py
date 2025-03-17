import sqlite3
import polars as pl

# Define file paths
csv_file_path = "output.csv"  # Processed CSV file
db_file_path = "unspsc.db"  # SQLite database file

# Load CSV into a Polars DataFrame
df = pl.read_csv(csv_file_path)

# Connect to SQLite database (or create if not exists)
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Define table schema (modify as needed based on CSV columns)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS unspsc (
        Version TEXT,
        Key INTEGER PRIMARY KEY,
        Segment_Code INTEGER,
        Segment_Title TEXT,
        Segment_Definition TEXT,
        Family_Code INTEGER,
        Family_Title TEXT,
        Family_Definition TEXT,
        Class_Code INTEGER,
        Class_Title TEXT,
        Class_Definition TEXT,
        Commodity_Code INTEGER,
        Commodity_Title TEXT,
        Commodity_Definition TEXT,
        Synonym TEXT,
        Acronym TEXT
    )
""")

# Insert data into the database
for row in df.to_dicts():
    cursor.execute("""
        INSERT INTO unspsc (
            Version, Key, Segment_Code, Segment_Title, Segment_Definition,
            Family_Code, Family_Title, Family_Definition, Class_Code, Class_Title, Class_Definition,
            Commodity_Code, Commodity_Title, Commodity_Definition, Synonym, Acronym
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("Version", ""),
        row.get("Key", ""),
        row.get("Segment", ""),
        row.get("Segment Title", ""),
        row.get("Segment Definition", ""),
        row.get("Family", ""),
        row.get("Family Title", ""),
        row.get("Family Definition", ""),
        row.get("Class", ""),
        row.get("Class Title", ""),
        row.get("Class Definition", ""),
        row.get("Commodity", ""),
        row.get("Commodity Title", ""),
        row.get("Commodity Definition", ""),
        row.get("Synonym", ""),
        row.get("Acronym", "")
    ))

# Commit and close connection
conn.commit()
conn.close()

print(f"Data successfully written to SQLite database: {db_file_path}")
