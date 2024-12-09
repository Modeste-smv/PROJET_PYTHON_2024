import pandas as pd
import yfinance as yf
from datetime import datetime

def process_expirations(symbols, min_date=None, max_date=None):
    """
    Télécharge les données d'options pour une liste de symboles donnés,
    en filtrant les dates d'expiration selon une plage minimale et maximale.

    Paramètres:
    - symbols (list of str): Liste des symboles des actifs (par exemple, ['AAPL', 'MSFT']).
    - min_date (datetime.date ou None): Date minimale d'expiration. Aucune limite si None.
    - max_date (datetime.date ou None): Date maximale d'expiration. Aucune limite si None.

    Retourne:
    - pandas.DataFrame: Les données d'options nettoyées avec les colonnes spécifiées.
    """
    data = pd.DataFrame()

    # Fonction interne pour filtrer les expirations selon min et max date
    def filter_expirations(available_exp):
        # available_exp est une liste de str au format 'YYYY-MM-DD'
        filtered = []
        for exp_str in available_exp:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if (min_date is None or exp_date >= min_date) and (max_date is None or exp_date <= max_date):
                filtered.append(exp_str)
        return filtered

    for sym in symbols:
        ticker_obj = yf.Ticker(sym)
        available_exp = ticker_obj.options  # liste de dates d'expiration (str)

        # Filtrer les expirations selon la plage de dates
        expirations_to_process = filter_expirations(available_exp)

        if not expirations_to_process:
            print(f"Aucune date d'expiration dans la plage fournie pour {sym}.")
            continue

        for expiration_date in expirations_to_process:
            try:
                options = ticker_obj.option_chain(expiration_date)
            except Exception as e:
                print(f"Erreur pour {sym} à la date {expiration_date}: {e}")
                continue

            # Séparation des calls et des puts
            calls = options.calls.copy()
            puts = options.puts.copy()

            # Ajout de la colonne 'optionType'
            calls['optionType'] = 'C'
            puts['optionType'] = 'P'

            # Fusion des données
            symbol_data = pd.concat([calls, puts], ignore_index=True)

            # Ajout du symbole et de la date d'expiration
            symbol_data['ticker'] = sym
            symbol_data['expiration_date'] = pd.to_datetime(expiration_date)

            # Sélection des colonnes d'intérêt
            symbol_data = symbol_data[['ticker','impliedVolatility', 'strike', 'bid', 'ask',  'expiration_date', 'optionType', 'lastPrice']]

            # Conversion des colonnes numériques
            numeric_cols = ['strike', 'bid', 'ask']
            symbol_data[numeric_cols] = symbol_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Suppression des lignes avec NaN critiques
            symbol_data.dropna(subset=['bid', 'ask'], inplace=True)

            # Réinitialisation de l'index
            symbol_data.reset_index(drop=True, inplace=True)

            # Concaténer dans le DataFrame global
            data = pd.concat([data, symbol_data], ignore_index=True)

    return data
