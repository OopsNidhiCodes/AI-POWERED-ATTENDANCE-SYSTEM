import sqlite3

conn = sqlite3.connect("faculty.db")
cursor = conn.cursor()

# Create table
cursor.execute("CREATE TABLE IF NOT EXISTS faculty (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")

# Insert sample faculty (Change the username/password as needed)
cursor.execute("INSERT INTO faculty (username, password) VALUES ('admin', 'admin123')")
conn.commit()
conn.close()

print("✅ Faculty database setup complete.")
