import sqlite3

DB_PATH = "db_prod_snapshot.sqlite"

def patch_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns_to_add = [
        ("source_url", "TEXT"),
        ("source_type", "TEXT"),
        ("parser_run_id", "TEXT"),
        ("parser_version", "TEXT"),
        ("last_parsed_at", "TIMESTAMP"),
        ("uds_storage_path", "TEXT"),
        ("contacts_phone", "TEXT"),
        ("contacts_email", "TEXT"),
        ("is_annual", "BOOLEAN"),
        ("audience", "TEXT")
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            print(f"Adding column {col_name}...")
            cursor.execute(f"ALTER TABLE festival ADD COLUMN {col_name} {col_type}")
            print(f"Added {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding {col_name}: {e}")
                
    conn.commit()
    conn.close()

if __name__ == "__main__":
    patch_db()
