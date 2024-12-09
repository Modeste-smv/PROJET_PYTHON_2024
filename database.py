import yfinance as yf
import sqlite3
from datetime import datetime

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
    TickerSymbol TEXT,
    ImpliedVolatility REAL,
    Strike REAL,
    Bid REAL,
    Ask REAL,
    ExpirationDate TEXT,
    OptionType TEXT,
    LastPrice REAL,
    FOREIGN KEY (TickerSymbol) REFERENCES Ticker (Symbol)
)
""")
conn.commit()

# Fonction pour insérer des données d'options dans la base
def insert_options_data(symbol):
    # Récupération des données avec yfinance
    ticker = yf.Ticker(symbol)
    options_dates = ticker.options

    # Ajout du ticker dans la table Ticker
    cursor.execute("INSERT OR IGNORE INTO Ticker (Symbol) VALUES (?)", (symbol,))
    conn.commit()

    for exp_date in options_dates:
        options_chain = ticker.option_chain(exp_date)
        for opt_type, options in [("call", options_chain.calls), ("put", options_chain.puts)]:
            for _, row in options.iterrows():
                cursor.execute("""
                INSERT INTO Options (
                    TickerSymbol, ImpliedVolatility, Strike, Bid, Ask, ExpirationDate, OptionType, LastPrice
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    row.get('impliedVolatility', None),
                    row['strike'],
                    row.get('bid', None),
                    row.get('ask', None),
                    exp_date,
                    opt_type,
                    row.get('lastPrice', None)
                ))
    conn.commit()

# Exemple d'utilisation
try:
    symbol = "AAPL"  # Exemple de symbole
    insert_options_data(symbol)
    print(f"Données pour le ticker {symbol} insérées avec succès.")
except Exception as e:
    print(f"Erreur : {e}")

# Fermeture de la connexion
conn.close()
