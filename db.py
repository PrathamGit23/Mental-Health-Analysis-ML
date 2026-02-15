import os
import sqlite3
import mysql.connector

def get_connection():

    # If deployed (Render/Cloud) → use SQLite
    if os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT"):
        conn = sqlite3.connect("database.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # Local development → use MySQL
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Pratham@23",   # keep your password
        database="mental_health_app"
    )