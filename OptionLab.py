import streamlit as st
import sqlite3
import datetime
import pandas as pd
from importation import process_expirations
import math
import scipy.stats as si
from scipy.stats import norm
import numpy as np

def monte_carlo_option_price(S, K, T, r, sigma, option_type='C', num_simulations=10000):
    """
    Calcule la valeur théorique d'une option selon la méthode de Monte Carlo.

    Paramètres:
    - S : Prix actuel du sous-jacent
    - K : Prix d'exercice (strike price)
    - T : Temps jusqu'à expiration (en années)
    - r : Taux sans risque
    - sigma : Volatilité implicite
    - option_type : 'C' pour Call, 'P' pour Put
    - num_simulations : Nombre de simulations Monte Carlo

    Retourne la valeur théorique de l'option calculée via Monte Carlo.
    """
    dt = T / 252  # Nombre de jours de trading par an
    discount_factor = np.exp(-r * T)  # Facteur de décote
    
    # Générer les chemins simulés
    simulated_prices = np.zeros(num_simulations)
    for i in range(num_simulations):
        price_path = S
        for t in range(int(T / dt)):
            price_path *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        simulated_prices[i] = price_path
    
    # Calculer la valeur de l'option
    if option_type == 'C':
        option_values = np.maximum(simulated_prices - K, 0)  # Call option payoff
    elif option_type == 'P':
        option_values = np.maximum(K - simulated_prices, 0)  # Put option payoff
    
    return discount_factor * np.mean(option_values)

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

    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Symbol FROM Ticker")
    symbols_in_db = [row[0] for row in cursor.fetchall()]
    conn.close()

    if symbols_in_db:
        symbol = st.selectbox("Sélectionnez un symbole d'action :", options=symbols_in_db)
    else:
        st.warning("Aucun symbole n'est disponible dans la base de données.")
        return

    # Choix de Call/Put
    option_type_display = st.selectbox("Type d'option :", options=["Call", "Put"])
    option_type_db = "C" if option_type_display == "Call" else "P"

    # Sélection de la date d'expiration précise
    expiration_date = st.date_input("Sélectionnez la date d'expiration :", value=datetime.date.today())

    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT strike 
        FROM Options 
        WHERE ticker = ? AND optionType = ? AND expiration_date = ?
        ORDER BY strike
    """, (symbol, option_type_db, expiration_date))
    strikes = [row[0] for row in cursor.fetchall()]
    conn.close()

    if strikes:
        strike_price = st.selectbox("Sélectionnez le strike price :", options=strikes)
    else:
        st.warning("Aucun strike price disponible pour cette option et cette date.")
        return

    st.write(f"Strike Price sélectionné : {strike_price}")
    st.write(f"Date d'expiration sélectionnée : {expiration_date}")

    data = st.session_state.get('options_data')
    selected_row = data[(data['ticker'] == symbol) & 
                        (data['optionType'] == option_type_db) & 
                        (data['expiration_date'] == expiration_date) & 
                        (data['strike'] == strike_price)].iloc[0]

    S = selected_row['underlying_price']
    K = selected_row['strike']
    T = (expiration_date - datetime.datetime.now().date()).days / 365
    r = 0.05  # Taux sans risque
    sigma = selected_row['implied_volatility']

    st.write(f"Prix du sous-jacent (S): {S}")
    st.write(f"Strike (K): {K}")
    st.write(f"Temps jusqu'à expiration (T): {T} ans")
    st.write(f"Taux sans risque (r): {r}")
    st.write(f"Volatilité implicite (σ): {sigma}")

    # Calculer le prix de l'option via Black-Scholes et Monte Carlo
    price_bs = black_scholes_price(S, K, T, r, sigma, option_type_db)
    price_mc = monte_carlo_option_price(S, K, T, r, sigma, option_type_db)

    st.write(f"Prix de l'option (Black-Scholes): {price_bs}")
    st.write(f"Prix de l'option (Monte Carlo): {price_mc}")

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
