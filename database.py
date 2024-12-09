import yfinance as yf
import sqlite3

# Création ou connexion à la base de données SQLite
conn = sqlite3.connect('options_data.db')
cursor = conn.cursor()

# Création des tables selon le modèle
cursor.execute("""
CREATE TABLE IF NOT EXISTS Ticker (
    Symbol TEXT PRIMARY KEY
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Options (
    ticker TEXT,
    impliedVolatility REAL,
    strike REAL,
    bid REAL,
    ask REAL,
    expiration_date TEXT,
    optionType TEXT,
    lastPrice REAL,
    FOREIGN KEY (ticker) REFERENCES Ticker (Symbol)
)
""")
conn.commit()
