# Importation des bibliothèques nécessaires
import pandas as pd
import yfinance as yf



def process_expirations(symbols, expiration_dates=None):
    """
    Télécharge les données d'options pour une ou plusieurs dates d'expiration
    et une liste de symboles donnés, puis les nettoie et retourne un DataFrame
    contenant uniquement les colonnes d'intérêt.

    Paramètres:
    - symbols (list of str): Liste des symboles des actifs (par exemple, ['AAPL', 'MSFT']).
    - expiration_dates (list of str): Liste des dates d'expiration au format '%Y-%m-%d'.

    Retourne:
    - pandas.DataFrame: Les données d'options nettoyées avec les colonnes spécifiées.
    """

    # DataFrame pour stocker les données de tous les symboles et dates d'expiration
    data = pd.DataFrame()

    for sym in symbols:  # Utiliser 'sym' au lieu de 'symbol' pour éviter toute confusion
        # Récupération des dates d'expiration disponibles pour le symbole
        ticker_obj = yf.Ticker(sym)
        available_exp = ticker_obj.options

        # Si expiration_dates est vide, utiliser toutes les dates disponibles
        if not expiration_dates:
            expirations_to_process = available_exp
        else:
            # Vérifier que les dates fournies sont disponibles pour le symbole
            expirations_to_process = [date for date in expiration_dates if date in available_exp]
            if not expirations_to_process:
                print(f"Aucune des dates d'expiration fournies n'est disponible pour {sym}.")
                continue

        for expiration_date in expirations_to_process:
            try:
                options = ticker_obj.option_chain(expiration_date)
            except Exception as e:
                print(f"Erreur pour {sym} à la date {expiration_date}: {e}")
                continue  # Passer à la date suivante en cas d'erreur

            # Séparation des calls et des puts
            calls = options.calls.copy()
            puts = options.puts.copy()

            # Ajout de la colonne 'optionType' pour distinguer les calls et les puts
            calls['optionType'] = 'C'
            puts['optionType'] = 'P'

            # Fusion des données de calls et puts
            symbol_data = pd.concat([calls, puts], ignore_index=True)

            # Ajout du symbole de l'actif
            symbol_data['ticker'] = sym

            # Ajout de la date d'expiration
            symbol_data['expiration_date'] = pd.to_datetime(expiration_date)

            # Sélection des colonnes d'intérêt
            symbol_data = symbol_data[['contractSymbol', 'impliedVolatility', 'strike', 'bid', 'ask', 'ticker', 'expiration_date', 'optionType', 'lastPrice']]

            # Conversion des colonnes numériques
            numeric_cols = ['strike', 'bid', 'ask']
            symbol_data[numeric_cols] = symbol_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Suppression des lignes avec des valeurs manquantes critiques
            symbol_data.dropna(subset=['bid', 'ask'], inplace=True)

            # Réinitialisation de l'index
            symbol_data.reset_index(drop=True, inplace=True)

            # Concaténer les données du symbole et de la date actuelle avec le DataFrame principal
            data = pd.concat([data, symbol_data], ignore_index=True)

    return data

