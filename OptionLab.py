import streamlit as st
import sqlite3
import datetime
import pandas as pd
from importation import process_expirations
import math
import scipy.stats as si
from scipy.stats import norm
import numpy as np



st.set_page_config(layout="wide")
# Définir les styles CSS pour la sidebar
page_bg_image = """
<style>
[data-testid="stSidebar"] {
    background-color: #0E3453;
}
[data-testid="stHeader"] {
    background-color: #A75502;
    padding-top: 10px;
}
[data-testid="collapsedControl"] {
    margin-top: -30px;
    align-items: center;
}
[data-testid="baseButton-headerNoPadding"] {
    padding-bottom: -30px;
}
[data-testid="stDeployButton"] {
    color: white;
}
[data-testid="baseButton-headerNoPadding"] {
    color: white;
}
[data-testid="stToolbar"] {
    color: white;
}

[data-testid="stSidebarHeader"] {
    background-color: #0E3453;
    padding: 0px;
    margin-left: 40px;
    text-align: center;
    align-items: center;
     margin-top: -10px;

}
[data-testid="stSidebar"] .sidebar-title {
    color: white;
    font-size: 18px;
    padding: 10px 15px;
    margin-bottom: 10px;
}
[data-testid="stSidebar"] .sidebar-section {
    padding: 10px;
}
.st-emotion-cache-5drf04 {
    height: 75px;
    width: 150px;
    z-index: 999990;
}
.st-emotion-cache-1kyxreq{
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    justify-content: center;
}
.stButton>button {
    background-color: #0E3453;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
    display: flex;
    align-items: center;
    margin: 5px 0;
    width: 100%;
    text-align: left;
    justify-content: flex-start;
}
.stButton>button:hover {
    background-color: #1F2A37;
}
.stButton>button:focus {
    outline: none;
}
.sidebar-icon {
    margin-right: 10px;
}
[data-testid="stMarkdownContainer"] {
    display: flex;
    justify-content: center;
}
</style>
"""

st.markdown(page_bg_image, unsafe_allow_html=True)
st.logo("image.png")

# Initialiser l'état de la page actuelle si nécessaire
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'accueil'

# Définition des boutons de navigation dans la barre latérale avec icônes
if st.sidebar.button('🏠 Accueil', key='accueil'):
    st.session_state.current_page = 'accueil'
if st.sidebar.button('💾 Données', key='donnees'):
    st.session_state.current_page = 'donnees'
if st.sidebar.button('📈 Pricing', key='pricing'):
    st.session_state.current_page = 'pricing'
if st.sidebar.button('📊 Sensibilités', key='sensibilites'):
    st.session_state.current_page = 'sensibilites'
if st.sidebar.button('🔍 Visualisation', key='visualisation'):
    st.session_state.current_page = 'visualisation'
if st.sidebar.button('⚖️ Comparaison', key='comparaison'):
    st.session_state.current_page = 'comparaison'
if st.sidebar.button('❓ Aide', key='aide'):
    st.session_state.current_page = 'aide'

# Fonction pour obtenir la page actuelle
def get_current_page():
    return st.session_state.current_page

def accueil():
    st.title('🏠 Accueil')
    st.write("Bienvenue dans l'application de pricing des options !")

def donnees():
    st.title('Données')
    st.write("Veuillez saisir les symboles (séparés par des virgules). Exemple : AAPL, MSFT, GOOGL")

    symbol_input = st.text_input("Symboles", value="AAPL")
    symbols = [sym.strip() for sym in symbol_input.split(",") if sym.strip()]

    st.write("Optionnel : Sélectionnez une plage de dates (min et max) pour filtrer les dates d'expiration disponibles.")
    min_date = st.date_input("Date minimale", value=None)
    max_date = st.date_input("Date maximale", value=None)

    if st.button("Importer les données"):
        if not symbols:
            st.warning("Veuillez saisir au moins un symbole.")
            return

        st.write("Récupération des données en cours...")
        data = process_expirations(symbols, min_date=min_date, max_date=max_date)
        if data.empty:
            st.warning("Aucune donnée disponible pour les symboles et la plage de dates fournie.")
        else:
            st.success(f"{len(data)} lignes récupérées.")
            st.dataframe(data)
            st.session_state['options_data'] = data
            st.session_state['symbols'] = symbols

            conn = sqlite3.connect('options_data.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Options")
            cursor.execute("DELETE FROM Ticker")
            conn.commit()

            unique_symbols = data['ticker'].unique()
            for sym in unique_symbols:
                cursor.execute("INSERT INTO Ticker (Symbol) VALUES (?)", (sym,))
            conn.commit()

            data.to_sql('Options', conn, if_exists='append', index=False)
            conn.close()

            st.success("Les données ont été insérées dans la base de données.")

def pricing():
    st.title('📈 Pricing')
    st.write("Calculez la valeur théorique de votre option.")

    # Connexion à la base de données
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()

    # Récupérer la liste des symboles uniques présents dans la base
    cursor.execute("SELECT DISTINCT Symbol FROM Ticker")
    symbols_in_db = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Étape 1 : Sélection du symbole d'action
    if symbols_in_db:
        symbol = st.selectbox("Sélectionnez un symbole d'action :", options=symbols_in_db)
    else:
        st.warning("Aucun symbole n'est disponible dans la base de données.")
        return

    # Étape 2 : Sélection du type d'option (Call ou Put)
    option_type_display = st.selectbox("Type d'option :", options=["Call", "Put"])
    option_type_db = "C" if option_type_display == "Call" else "P"

    # Connexion à la base pour récupérer les expirations
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT DISTINCT strftime('%Y', expiration_date) AS year
        FROM Options 
        WHERE ticker = ? AND optionType = ? 
        ORDER BY year
    """, (symbol, option_type_db))
    years = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Étape 3 : Filtrage par année
    if years:
        selected_year = st.selectbox("Sélectionnez l'année d'expiration :", options=years)
    else:
        st.warning("Aucune date d'expiration disponible pour ce symbole et ce type d'option.")
        return

    # Connexion pour récupérer les mois
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT DISTINCT strftime('%m', expiration_date) AS month
        FROM Options 
        WHERE ticker = ? AND optionType = ? AND strftime('%Y', expiration_date) = ?
        ORDER BY month
    """, (symbol, option_type_db, selected_year))
    months = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Étape 4 : Filtrage par mois
    if months:
        selected_month = st.selectbox("Sélectionnez le mois d'expiration :", options=months)
    else:
        st.warning("Aucune date d'expiration disponible pour cette année sélectionnée.")
        return

    # Connexion pour récupérer les jours
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT DISTINCT expiration_date
        FROM Options 
        WHERE ticker = ? AND optionType = ? 
        AND strftime('%Y', expiration_date) = ? 
        AND strftime('%m', expiration_date) = ?
        ORDER BY expiration_date
    """, (symbol, option_type_db, selected_year, selected_month))
    days = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Étape 5 : Filtrage par jour
    if days:
        selected_day = st.selectbox("Sélectionnez la date d'expiration :", options=days)
    else:
        st.warning("Aucune date d'expiration disponible pour ce mois sélectionné.")
        return

    # Connexion pour récupérer les strikes
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT DISTINCT strike
        FROM Options 
        WHERE ticker = ? AND optionType = ? AND expiration_date = ?
        ORDER BY strike
    """, (symbol, option_type_db, selected_day))
    strikes = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Étape 6 : Filtrage par prix d'exercice (Strike)
    if strikes:
        selected_strike = st.selectbox("Sélectionnez le prix d'exercice (strike) :", options=strikes)
    else:
        st.warning("Aucun strike disponible pour cette date sélectionnée.")
        return

    # Récupération du dernier prix de l'option
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT lastPrice
        FROM Options 
        WHERE ticker = ? AND optionType = ? AND expiration_date = ? AND strike = ?
    """, (symbol, option_type_db, selected_day, selected_strike))
    last_price = cursor.fetchone()
    conn.close()

    if last_price is None:
        st.warning("Aucun dernier prix disponible pour cette option.")
        return

    last_price = last_price[0]  # Dernier prix auquel l'option s'est vendue

    # Récupération du prix du sous-jacent, volatilité et taux sans risque
    # Pour l'exemple, prenons des valeurs hypothétiques
    S = 150  # Prix du sous-jacent (à ajuster selon le symbole)
    sigma = 0.25  # Volatilité (à ajuster selon les données)
    r = 0.05  # Taux d'intérêt sans risque (à ajuster selon le contexte)
    T = (pd.to_datetime(selected_day) - pd.to_datetime(datetime.date.today())).days / 365.0

    # Calcul de la valeur théorique de l'option avec Monte Carlo
    option_value = monte_carlo_option_pricing(S, selected_strike, T, r, sigma, option_type_db)

    # Affichage de la comparaison
    st.write(f"### Comparaison de la valeur théorique avec le dernier prix")
    st.write(f"**Valeur théorique de l'option via Monte Carlo** : {option_value:.2f}")
    st.write(f"**Dernier prix de l'option** : {last_price:.2f}")

    # Calcul du ratio entre la valeur théorique et le dernier prix
    ratio = option_value / last_price

    # Comparaison et recommandation pour le vendeur
    if ratio > 1:
        st.write("La valeur théorique de l'option est supérieure au dernier prix de vente.")
        st.write("Cela suggère que l'option est sous-évaluée sur le marché. Il pourrait être avantageux pour vous de vendre cette option.")

def sensibilites():
    st.title('📊 Sensibilités')
    st.write("Calcul des sensibilités des options (Greeks).")

def visualisation():
    st.title('🔍 Visualisation')
    st.write("Visualisation des données et des résultats.")

def comparaison():
    st.title('⚖️ Comparaison')
    st.write("Comparez les prix de marché aux prix théoriques calculés.")

    # Ajoutez la logique de comparaison des options si vous le souhaitez
    st.write("Comparer les résultats ici.")

def aide():
    st.title('❓ Aide')
    st.write("Consultez l'aide pour plus d'informations.")

# Associer les pages à leurs fonctions respectives
functions = {
    "accueil": accueil,
    "donnees": donnees,
    "pricing": pricing,
    "sensibilites": sensibilites,
    "visualisation": visualisation,
    "comparaison": comparaison,
    "aide": aide,
}


# Afficher la page sélectionnée
current_page = get_current_page()
if current_page in functions:
    functions[current_page]()
else:
    st.write("Page non trouvée.")
