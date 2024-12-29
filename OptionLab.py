import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from importation import process_expirations
import s3fs
import yfinance as yf
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ta
import os
import base64
from PIL import Image




# Liaison à la base
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'})
data = pd.read_parquet("s3://modestesmv/database.parquet", filesystem=fs)
# Conversion explicite de la colonne expiration_date en datetime
data['expiration_date'] = pd.to_datetime(data['expiration_date'], errors='coerce')


################################################### FONCTIONS PRÉLIMINAIRES ######################################################

# Simulation de Monte Carlo 
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


# Récupération des taux sans risque des bons du trésor depuis Yahoo Finance
def get_risk_free_rates():
    tickers = {
        "Bon du trésor américain à 13 semaines": "^IRX",
        "Bon du trésor américain à 5 ans": "^FVX",
        "Bon du trésor américain à 10 ans": "^TNX",
        "Bon du trésor américain à 30 ans": "^TYX"
    }
    rates = {}
    for name, ticker in tickers.items():
        data = yf.download(ticker, period="1d", interval="1d")
        if not data.empty:
            rates[name] = data["Close"].iloc[-1] / 100  # Convertir en taux décimal
    return rates

##################################### PARAMETRES DE MISE EN FORME ET DES STYLES ##############################################

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

ticker_names = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "XOM": "ExxonMobil",
    "PG": "Procter & Gamble",
    "NVDA": "NVIDIA"
}

def custom_styling(df):
    # Style pour le header
    header_props = [('background-color', '#0E3453'),  # Blue background
                    ('color', 'white'),               # White text
                    ('border', '1px solid black'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')]

    # Style pour le corps
    body_props = [('background-color', '#DCE6F1'),  # Light blue background
                  ('color', 'black'),                # Black text
                  ('border', '1px solid black'),
                  ('text-align', 'center')]

    return df.style.set_table_styles([
        {'selector': 'thead th', 'props': header_props},
        {'selector': 'tbody td', 'props': body_props}
    ]).hide(axis="index")


st.markdown(page_bg_image, unsafe_allow_html=True)

# Afficher l'image du logo dans la section de l'en-tête de la sidebar
st.logo("media/image.png")

# Initialiser l'état de la page actuelle si nécessaire
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'accueil'

# Définition des boutons de navigation dans la barre latérale avec icônes
if st.sidebar.button('🏠 Accueil', key='accueil'):
    st.session_state.current_page = 'accueil'
if st.sidebar.button('💾 Données', key='donnees'):
    st.session_state.current_page = 'donnees'
if st.sidebar.button('🔍 Visualisation', key='visualisation'):
    st.session_state.current_page = 'visualisation'
if st.sidebar.button('📈 Pricing', key='pricing'):
    st.session_state.current_page = 'pricing'
if st.sidebar.button('📊 Sensibilités', key='sensibilites'):
    st.session_state.current_page = 'sensibilites'
if st.sidebar.button('📖 Documentation', key='documentation'):
    st.session_state.current_page = 'documentation'


# Fonction pour obtenir la page actuelle
def get_current_page():
    return st.session_state.current_page


##################################################### PAGE D'ACCUEIL #############################################################

# Définition des fonctions pour chaque page
def accueil():
    # Titre principal et sous-titre
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color:#0E3453;'>OptionLab</h1>
            <h4 style='color:#555555;'>Votre assistant complet pour l'analyse et la gestion des options</h4>
        </div>
    """, unsafe_allow_html=True)

    # Présentation générale de l'application
    st.markdown("""
        <p style='text-align: center; color:#A75502;'>
            Bienvenue dans <b>OptionLab</b>, une application conçue pour répondre aux besoins des acheteurs, des vendeurs, et des analystes de marché. 
            Elle vous offre une plateforme pour explorer, analyser et gérer les options. 
        </p>
    """, unsafe_allow_html=True)

    # Vidéo de présentation
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col1:
        st.write("")
    with col2:
        st.video('media/video_intro.mp4')
    with col3:
        st.write("")

    # Aide et support
    st.markdown("""
        <p style='text-align: justify; color:#333333;'>Nous espérons que <b>OptionLab</b> sera un outil précieux pour vos analyses et décisions.
        </p>
    """, unsafe_allow_html=True)


########################################### CONSULTER LES OPTIONS DISPONIBLES ####################################################

def donnees():
    st.markdown("""
    <div style='text-align: center;'> <h1 style='color:#0E3453;'>Options Disponibles</h1></div>
    """, unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:15px;color:#A75502'>Consultez les informations sur les options disponibles dans la base de données.</h6>",unsafe_allow_html=True)


    # Disposer les filtres sur une même ligne
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        # Récupérer les tickers uniques
        tickers = data['ticker'].unique()
        selected_ticker = st.selectbox("Sélectionnez un ticker", options=sorted(tickers))

    with col2:
        # Sélection du type d'option
        option_type = st.selectbox("Type d'option", options=['Call', 'Put'])

    with col3:
        # Sélection de l'année d'expiration
        data['Year'] = pd.to_datetime(data['expiration_date']).dt.year
        years = data[data['ticker'] == selected_ticker]['Year'].unique()
        selected_year = st.selectbox("Année", options=sorted(years))

    with col4:
        # Sélection du mois d'expiration
        data['Month'] = pd.to_datetime(data['expiration_date']).dt.month
        months = data[(data['ticker'] == selected_ticker) & (data['Year'] == selected_year)]['Month'].unique()
        selected_month = st.selectbox("Mois", options=sorted(months))

    with col5:
        # Sélection du jour d'expiration
        data['Day'] = pd.to_datetime(data['expiration_date']).dt.day
        days = data[(data['ticker'] == selected_ticker) & 
                    (data['Year'] == selected_year) & 
                    (data['Month'] == selected_month)]['Day'].unique()
        selected_day = st.selectbox("Jour", options=sorted(days))

    with col6:
        # Filtre sur les strikes avec des tranches
        strikes = data[(data['ticker'] == selected_ticker) & 
                       (data['Year'] == selected_year) & 
                       (data['Month'] == selected_month) & 
                       (data['Day'] == selected_day)]['strike']
        min_strike, max_strike = strikes.min(), strikes.max()
        selected_strike_range = st.slider("Tranche de strikes", 
                                          min_value=float(min_strike), 
                                          max_value=float(max_strike), 
                                          value=(float(min_strike), float(max_strike)))

    # Filtrer les données selon les critères sélectionnés
    filtered_data = data[(data['ticker'] == selected_ticker) &
                         (data['optionType'] == option_type) &
                         (data['Year'] == selected_year) &
                         (data['Month'] == selected_month) &
                         (data['Day'] == selected_day) &
                         (data['strike'] >= selected_strike_range[0]) &
                         (data['strike'] <= selected_strike_range[1])]

    # Suppression des colonnes inutiles
    filtered_data = filtered_data.drop(columns=['ticker', 'Year', 'Month', 'Day', 'expiration_date'])

    # Réarranger l'ordre des colonnes
    column_order = ['optionType', 'strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility']
    filtered_data = filtered_data[column_order]

    # Arrondi des colonnes numériques à deux chiffres sauf impliedVolatility
    for col in filtered_data.columns:
        if col != 'impliedVolatility' and filtered_data[col].dtype == 'float64':
            filtered_data[col] = filtered_data[col].apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if x % 1 != 0 else int(x))

    # Affichage des données filtrées avec stylisation
    if filtered_data.empty:
        st.warning("Aucune donnée correspondante trouvée.")
    else:
        st.markdown(f"<h5 style='text-align: center;margin-top: 40px'>Options disponibles pour {ticker_names.get(selected_ticker, selected_ticker)} expirant le {selected_year}-{selected_month:02d}-{selected_day:02d}</h5>", unsafe_allow_html=True)
        styled_df = custom_styling(filtered_data)
        st.markdown(styled_df.to_html(), unsafe_allow_html=True)


################################################### PRICING DES OPTIONS ##########################################################

######################### Partie 1 : Definition de la classe pour la methode LSMC sur une option #################################

class AmericanOptionsLSMC:
    """Classe pour valoriser les options américaines et calculer les Greeks en utilisant Longstaff-Schwartz (2001)."""

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations):
        self.option_type = option_type.lower()
        self.S0 = float(S0)
        self.strike = float(strike)
        self.T = float(T)
        self.M = int(M)
        self.r = float(r)
        self.div = float(div)
        self.sigma = float(sigma)
        self.simulations = int(simulations)

        if self.option_type not in ['call', 'put']:
            raise ValueError("Le type d'option doit être 'call' ou 'put'.")

        self.time_unit = self.T / self.M
        self.discount = np.exp(-self.r * self.time_unit)

        # Variables pour stocker les trajectoires
        self.price_paths = None
        self.payoffs = None

    def _generate_price_paths(self, seed=123):
        """Génère les trajectoires de prix via Monte Carlo."""
        np.random.seed(seed)
        prices = np.zeros((self.M + 1, self.simulations))
        prices[0, :] = self.S0

        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = np.concatenate((brownian, -brownian))
            prices[t, :] = prices[t - 1, :] * np.exp(
                (self.r - self.sigma ** 2 / 2) * self.time_unit + self.sigma * np.sqrt(self.time_unit) * brownian
            )
        self.price_paths = prices
        return prices

    def _calculate_payoffs(self):
        """Calcule les payoffs selon les trajectoires générées."""
        if self.price_paths is None:
            raise ValueError("Les trajectoires de prix doivent être générées avant de calculer les payoffs.")

        if self.option_type == 'call':
            self.payoffs = np.maximum(self.price_paths - self.strike, 0)
        elif self.option_type == 'put':
            self.payoffs = np.maximum(self.strike - self.price_paths, 0)
        return self.payoffs

    def _backward_induction(self):
        """Effectue la régression et le retour en arrière pour calculer la valeur."""
        if self.payoffs is None:
            raise ValueError("Les payoffs doivent être calculés avant de faire le retour en arrière.")

        values = np.zeros_like(self.payoffs)
        values[-1, :] = self.payoffs[-1, :]

        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(self.price_paths[t, :], values[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.price_paths[t, :])
            values[t, :] = np.where(self.payoffs[t, :] > continuation_value, self.payoffs[t, :], values[t + 1, :] * self.discount)

        return values[1, :] * self.discount

    def price(self):
        """Calcule le prix de l'option."""
        if self.price_paths is None:
            self._generate_price_paths()
        if self.payoffs is None:
            self._calculate_payoffs()
        values = self._backward_induction()
        return np.mean(values)

    # Méthodes pour calculer les Greeks
    def delta(self, h=0.01):
        """Calcule Delta via différences finies."""
        S_up = self.S0 + h
        S_down = self.S0 - h
        option_up = AmericanOptionsLSMC(self.option_type, S_up, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        option_down = AmericanOptionsLSMC(self.option_type, S_down, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        return (option_up.price() - option_down.price()) / (2 * h)

    def gamma(self, h=0.01):
        """Calcule Gamma via différences finies."""
        S_up = self.S0 + h
        S_down = self.S0 - h
        option_up = AmericanOptionsLSMC(self.option_type, S_up, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        option_down = AmericanOptionsLSMC(self.option_type, S_down, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        option_center = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        return (option_up.price() - 2 * option_center.price() + option_down.price()) / (h ** 2)

    def vega(self, h=0.01):
        """Calcule Vega via différences finies."""
        sigma_up = self.sigma + h
        sigma_center = self.sigma
        option_up = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, sigma_up, self.simulations)
        option_center = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, sigma_center, self.simulations)
        return (option_up.price() - option_center.price()) / (2 * h)

    def rho(self, h=0.01):
        """Calcule Rho via différences finies."""
        r_up = self.r + h
        r_center = self.r
        option_up = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, r_up, self.div, self.sigma, self.simulations)
        option_center = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, r_center, self.div, self.sigma, self.simulations)
        return (option_up.price() - option_center.price()) / (2 * h)

    def theta(self, h=1/365):
        """Calcule Theta via différences finies."""
        T_up = self.T - h
        T_center = self.T
        option_up = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, T_up, self.M, self.r, self.div, self.sigma, self.simulations)
        option_center = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, T_center, self.M, self.r, self.div, self.sigma, self.simulations)
        return (option_up.price() - option_center.price()) / h


##################################################################################################################################
############################# Partie 2 : Fonction pour effectuer le pricing a la base des formulaires ############################
##################################################################################################################################

def pricing():
    st.markdown("""<div style='text-align: center;'> <h1 style='color:#0E3453;'>Pricing des Options</h1></div>""", unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:15px;color:#A75502'>Calculez ou modélisez les valeurs théoriques des options selon votre rôle.</h6>",unsafe_allow_html=True)

    # Vérifier si la base de données est chargée
    if data.empty:
        st.warning("La base de données est vide. Veuillez vérifier la source.")
        return

    # Récupérer les symboles uniques disponibles
    symbols_in_db = data['ticker'].unique().tolist()

    # Utilisation d'un sélecteur unique pour choisir le rôle
    role = st.radio(
        "Veuillez sélectionner votre rôle :",
        options=["Acheteur", "Vendeur"],
        index=0,
        horizontal=True
    )

    st.write(f"Rôle sélectionné : **{role}**")

    if role == "Vendeur":
        # Si c'est un vendeur, on demande les paramètres nécessaires
        
        # Sélection du ticker
        if symbols_in_db:
            symbol = st.selectbox("Symbole de l'actif", options=symbols_in_db)
        else:
            st.warning("Aucun symbole n'est disponible dans la base de données.")
            return
            
        # Choix du type d'option
        option_type = st.selectbox("Type d'option :", options=["Call", "Put"])

        # Filtrer les données pour récupérer les dates d'expirations disponibles
        df_symbol = data[(data['ticker'] == symbol) & (data['optionType'] == option_type)]
        if df_symbol.empty:
            st.warning(f"Aucune donnée disponible pour {symbol} ({option_type}).")
            return

        # Extraire et trier les dates d'expiration disponibles
        df_symbol['expiration_date'] = pd.to_datetime(df_symbol['expiration_date'])
        df_exp = df_symbol[['expiration_date']].drop_duplicates().sort_values('expiration_date')

        # Sélection des années
        df_exp['Year'] = df_exp['expiration_date'].dt.year
        years = sorted(df_exp['Year'].unique())
        selected_year = st.selectbox("Année d'expiration", options=years)

        # Filtrer par année
        df_year = df_exp[df_exp['Year'] == selected_year]
        df_year['Month'] = df_year['expiration_date'].dt.month
        months = sorted(df_year['Month'].unique())
        selected_month = st.selectbox("Mois d'expiration", options=months, format_func=lambda m: f"{m:02d}")

        # Filtrer par mois
        df_month = df_year[df_year['Month'] == selected_month]
        available_dates = sorted(df_month['expiration_date'].dt.strftime('%Y-%m-%d').unique())
        expiration_date = st.selectbox("Date d'expiration", options=available_dates)

        # Saisie du strike price (prix d'exercice)
        df_strike = df_symbol[df_symbol['expiration_date'] == expiration_date]
        all_strikes = sorted(df_strike['strike'].unique())

        if not all_strikes:
            st.warning("Aucun prix d'exercice disponible pour cette date d'expiration.")
            return

        # Si les strikes sont nombreux et réguliers, on peut utiliser un slider
        # Sinon, un selectbox est plus approprié
        # Exemple avec un selectbox (plus sûr si irrégulier) :
        strike_price = st.selectbox("Prix d'exercice (Strike price)", options=all_strikes)

        selected_row = df_symbol[
            (df_symbol['expiration_date'] == expiration_date) &
            (df_symbol['strike'] == strike_price) &
            (df_symbol['optionType'] == option_type)
        ]
        selected_row = selected_row.iloc[0]  # Récupère la première ligne
    
        # Bouton pour calculer la valeur théorique
        if st.button("Calculer la valeur de l'option"):
            # Récupérer les données pour le ticker
            ticker = yf.Ticker(symbol)
            S0 = ticker.history(period="1d")['Close'].iloc[-1]
            K = selected_row['strike']
            T = (pd.to_datetime(selected_row['expiration_date'], unit='ms') - pd.Timestamp.now()).days / 365.0
            r = risk_free_rates['Bon du trésor américain à 5 ans']  # Choix arbitraire parmi les choix de taux sans risque possibles
            sigma = selected_row['impliedVolatility']
            M = 50  # Nombre de pas
            simulations = 10000
            # Calcul du prix théorique de l'option
            american_option = AmericanOptionsLSMC(option_type.lower(), S0, K, T, M, r, 0, sigma, simulations)
            option_price = american_option.price()

            # Mise en forme visuelle des résultats
            last_price=selected_row['lastPrice']
            if last_price > option_price:
                conseil = f"Nous vous conseillons de <b style='color:#A75502;'>vendre cette option</b>, car son dernier prix de marché (<b>{last_price:.2f} €</b>) est supérieur à sa valeur théorique estimée."
            else:
                conseil = f"Cette option semble sous-évaluée. Il pourrait être préférable d'attendre une augmentation de son prix de marché."
            resultat_html = f"""
            <div style="text-align:center; margin-top:20px; padding:20px; background-color:#f9f9f9; border: 1px solid #ddd; border-radius:15px;">
                <h4 style="color:#A75502; font-size:20px; font-weight:bold; margin-bottom:10px;">Résultat de l'estimation</h4>
                <p style="font-size:15px; color:#0E3453; line-height:1.5;">
                    La valeur théorique de l'option américaine est estimée à <b style="color:#A75502;">{option_price:.2f} €</b>.
                </p>
                <p style="font-size:15px; color:#0E3453; line-height:1.5;">
                    {conseil}
                </p>
                <p style="font-size:15px; color:#0E3453; line-height:1.5;">
                    Rappel : Le dernier prix de marché enregistré est de <b style="color:#0E3453;font-weight:bold">{last_price:.2f} €</b>.
                </p>
            </div>
            """
            st.markdown(resultat_html, unsafe_allow_html=True)
    
    elif role == "Acheteur":
        # Si c'est un acheteur, on demande les paramètres nécessaires

        # Sélection du ticker
        if symbols_in_db:
            symbol = st.selectbox("Choisissez un symbole d'actif", options=symbols_in_db)
        else:
            st.warning("Aucun symbole n'est disponible dans la base de données.")
            return

        # Choix du type d'option
        option_type = st.selectbox("Type d'option :", options=["Call", "Put"])

        # Filtrer les données pour récupérer les dates d'expirations disponibles
        df_symbol2 = data[(data['ticker'] == symbol) & (data['optionType'] == option_type)]
        if df_symbol2.empty:
            st.warning(f"Aucune donnée disponible pour {symbol} ({option_type}).")
            return

        # Extraire et trier les dates d'expiration disponibles
        df_symbol2['expiration_date'] = pd.to_datetime(df_symbol2['expiration_date'])
        df_exp = df_symbol2[['expiration_date']].drop_duplicates().sort_values('expiration_date')

        # Sélection des années
        df_exp['Year'] = df_exp['expiration_date'].dt.year
        years = sorted(df_exp['Year'].unique())
        selected_year = st.selectbox("Année d'expiration", options=years)

        # Filtrer par année
        df_year = df_exp[df_exp['Year'] == selected_year]
        df_year['Month'] = df_year['expiration_date'].dt.month
        months = sorted(df_year['Month'].unique())
        selected_month = st.selectbox("Mois d'expiration", options=months, format_func=lambda m: f"{m:02d}")

        # Filtrer par mois
        df_month = df_year[df_year['Month'] == selected_month]
        available_dates = sorted(df_month['expiration_date'].dt.strftime('%Y-%m-%d').unique())
        expiration_date = st.selectbox("Date d'expiration", options=available_dates)

        # Filtrer les options disponibles pour la date choisie
        df_filtered = df_symbol2[df_symbol2['expiration_date'] == expiration_date]

        # Bouton pour modéliser les valeurs des options
        if st.button("Modéliser les valeurs des options"):

            # Graphique : Valeurs théoriques et derniers prix en fonction du strike price
            if not df_filtered.empty:
                strikes = df_filtered['strike'].unique()
                theoretical_values = []
                last_prices = []

                # Calcul des valeurs théoriques pour chaque strike
                for _, row in df_filtered.iterrows():
                    S0 = yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1]
                    K = row['strike']
                    T = (pd.to_datetime(row['expiration_date']) - pd.Timestamp.now()).days / 365.0
                    r = risk_free_rates['Bon du trésor américain à 5 ans']  # Choix arbitraire parmi les choix de taux sans risque possibles
                    sigma = row['impliedVolatility']
                    M = 50  # Nombre de pas
                    simulations = 10000

                    american_option = AmericanOptionsLSMC(option_type.lower(), S0, K, T, M, r, 0, sigma, simulations)
                    theoretical_values.append(american_option.price())
                    last_prices.append(row['lastPrice'])

                # Création du graphique interactif avec Plotly
                fig = go.Figure()

                # Tracer les courbes des valeurs théoriques et des derniers prix en fonction du strike price
                fig.add_trace(go.Scatter(
                    x=strikes,
                    y=theoretical_values,
                    mode='lines+markers',
                    name="Valeur théorique",
                    line=dict(color='#0E3453'),
                    hovertemplate='Strike price : %{x}<br>Valeur théorique : %{y:.2f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=strikes,
                    y=last_prices,
                    mode='lines+markers',
                    name="Valeur de marché",
                    line=dict(color='#A75502'),
                    hovertemplate='Strike price : %{x}<br>Valeur de marché : %{y:.2f}<extra></extra>'
                ))

                # Mise en forme du graphique
                fig.update_layout(
                    title=f"{symbol} ({option_type}) - {expiration_date}",
                    xaxis_title="Prix d'exercice (Strike price)",
                    yaxis_title="Valeur",
                    legend=dict(font=dict(size=10)),
                    font=dict(size=10),
                    hovermode="x unified"
                )

                # Affichage du graphique interactif
                st.plotly_chart(fig, use_container_width=True)

                # Calcul de l'option à conseiller
                differences = [(tv - lp) for tv, lp in zip(theoretical_values, last_prices)]
                max_diff_index = differences.index(max(differences))
                best_strike = strikes[max_diff_index]
                best_theoretical_value = theoretical_values[max_diff_index]
                best_last_price = last_prices[max_diff_index]

                # Message de recommandation
                conseil_html = f"""
                <div style="text-align:center; margin-top:20px; padding:20px; background-color:#f9f9f9; border: 1px solid #ddd; border-radius:15px;">
                    <h4 style="color:#A75502; font-size:20px; font-weight:bold; margin-bottom:10px;">Recommandation d'Option</h4>
                    <p style="font-size:15px; color:#0E3453; line-height:1.5;">
                        L'option avec le <b>prix d'exercice (strike)</b> de <b style="color:#A75502;">{best_strike}</b> expirant le <b style="color:#A75502;">{expiration_date}</b> est fortement recommandée.<br>
                        Cette option est la plus sous-évaluée, avec une valeur théorique de <span style="color:#0E3453;font-weight:bold">{best_theoretical_value:.2f}</span> contre un dernier prix de marché de <span style="color:#A75502;font-weight:bold">{best_last_price:.2f}</span>.
                    </p>
                </div>
                """
                st.markdown(conseil_html, unsafe_allow_html=True)


    else:
        st.warning("Aucune option disponible pour cette date d'expiration.")


################################################# ANALYSE DES SENSIBILITES #######################################################

def sensibilites():
    st.markdown("""
    <div style='text-align: center;'><h1 style='color:#0E3453;'>Analyse des Sensibilités</h1></div>
    """, unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:15px;color:#A75502'>Évaluez les paramètres clés (Greeks) pour une meilleure compréhension de vos options.</h6>", unsafe_allow_html=True)

    # Sélection du ticker
    tickers = data['ticker'].unique()
    selected_ticker = st.selectbox("Sélectionnez un ticker", tickers)

    # Sélection du type d'option
    option_type = st.selectbox("Type d'option :", options=["Call", "Put"])

    # Filtrer les données pour récupérer les dates d'expirations disponibles
    df_symbol = data[(data['ticker'] == selected_ticker) & (data['optionType'] == option_type)]
    if df_symbol.empty:
        st.warning(f"Aucune donnée disponible pour {selected_ticker} ({option_type}).")
        return

    # Sélection de l'année d'expiration
    df_symbol['expiration_date'] = pd.to_datetime(df_symbol['expiration_date'])
    df_symbol['Year'] = df_symbol['expiration_date'].dt.year
    years = sorted(df_symbol['Year'].unique())
    selected_year = st.selectbox("Sélectionnez l'année d'expiration", years)

    # Filtrer par année
    df_year = df_symbol[df_symbol['Year'] == selected_year]

    # Sélection du mois d'expiration
    df_year['Month'] = df_year['expiration_date'].dt.month
    months = sorted(df_year['Month'].unique())
    selected_month = st.selectbox("Sélectionnez le mois d'expiration", months, format_func=lambda m: f"{m:02d}")

    # Filtrer par mois
    df_month = df_year[df_year['Month'] == selected_month]

    # Sélection des dates exactes d'expiration
    available_dates = sorted(df_month['expiration_date'].dt.strftime('%Y-%m-%d').unique())
    selected_expiration = st.selectbox("Sélectionnez une date d'expiration", available_dates)

    # Filtrer les strikes disponibles
    df_filtered = df_month[df_month['expiration_date'] == selected_expiration]
    strikes = sorted(df_filtered['strike'].unique())
    selected_strike = st.selectbox("Sélectionnez un prix d'exercice (strike)", strikes)

    # Vérification des données correspondantes
    if df_filtered.empty:
        st.error("Aucune donnée correspondante trouvée. Veuillez ajuster vos sélections.")
        return

    # Définir les autres paramètres
    r = risk_free_rates['Bon du trésor américain à 5 ans']  # Choix arbitraire parmi les choix de taux sans risque possibles
    N = st.number_input("Nombre de trajectoires Monte Carlo (N)", value=10000, step=1000)
    M = st.number_input("Nombre de pas dans la simulation (M)", value=50, step=10)

    # Paramètres pour le calcul
    S0 = data.loc[data['ticker'] == selected_ticker, 'lastPrice'].iloc[0]
    T = (pd.to_datetime(selected_expiration) - pd.Timestamp.now()).days / 365.0
    K = selected_strike
    sigma = df_filtered['impliedVolatility'].iloc[0]

    # Créer une instance de la classe AmericanOptionsLSMC
    option = AmericanOptionsLSMC(
        option_type=option_type.lower(),
        S0=S0,
        strike=K,
        T=T,
        M=int(M),
        r=r,
        div=0.0,  # Supposons un dividende nul
        sigma=sigma,
        simulations=int(N)
    )
    # Tableau récapitulatif des sensibilités
    greek_data = {
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Rôle": [
            "Sensibilité du prix de l'option à une variation du prix de l'actif sous-jacent.",
            "Variation de Delta en réponse à une variation du prix de l'actif sous-jacent.",
            "Sensibilité du prix de l'option à une variation de la volatilité implicite.",
            "Sensibilité du prix de l'option au passage du temps.",
            "Sensibilité du prix de l'option à une variation du taux d'intérêt sans risque."
        ],
        "Estimation": [
            "-",
            "-",
            "-",
            "-",
            "-"
        ]
    }
    df_greeks = pd.DataFrame(greek_data)
    # Calculer les sensibilités
    if st.button("Calculer"):
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega()
        theta =  option.theta()
        rho = option.rho()

        # Mettre à jour le tableau avec les estimations
        df_greeks["Estimation"] = [f"{delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{theta:.4f}", f"{rho:.4f}"]
    # Afficher le tableau
    styled_df = custom_styling(df_greeks)
    st.markdown(styled_df.to_html(), unsafe_allow_html=True)


################################################# VISUALISATION EN TEMPS RÉEL ###################################################

######################### Partie 1 : Fonctions pour Récupérer, Traiter et Créer des Indicateurs Techniques ######################

# Récupérer les données boursières
def recuperer_donnees_boursieres(ticker, periode, intervalle):
    date_fin = datetime.now()
    if periode == '1wk':
        date_debut = date_fin - timedelta(days=7)
        data = yf.download(ticker, start=date_debut, end=date_fin, interval=intervalle)
        data.columns = data.columns.get_level_values(0)
    else:
        data = yf.download(ticker, period=periode, interval=intervalle)
        data.columns = data.columns.get_level_values(0)
    return data

# Traitement des données boursières
def traiter_donnees(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Calcul des métriques principales
def calculer_metriques(data):
    dernier_cours = data['Close'].iloc[-1]
    cours_precedent = data['Close'].iloc[0]
    variation = dernier_cours - cours_precedent
    variation_pourcentage = (variation / cours_precedent) * 100
    haut = data['High'].max()
    bas = data['Low'].min()
    volume = data['Volume'].sum()
    return dernier_cours, variation, variation_pourcentage, haut, bas, volume

# Ajout des indicateurs 
def ajouter_indicateurs_techniques(data):
    data['MMS_20'] = ta.trend.sma_indicator(data["Close"], window=20)
    data['MME_20'] = ta.trend.ema_indicator(data["Close"], window=20)
    return data


###################################### PART 2: Création de la Mise en Page de Visualisation #####################################

# Fonction principale pour l'onglet visualisation
def visualisation():
    st.markdown("""
    <div style='text-align: center;'> <h1 style='color:#0E3453;'>Visualisation des Marchés Financiers</h1></div>
    """, unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:15px;color:#A75502'>Visualisez les fluctuations des prix et les indicateurs techniques pour les actions sélectionnées.</h6>",unsafe_allow_html=True)

    # Création des colonnes pour le formulaire
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker = st.selectbox('Ticker', options=['AAPL', 'MSFT', 'TSLA', 'JNJ', 'JPM', 'XOM', 'PG', 'NVDA'])

    with col2:
        periode = st.selectbox('Période', options=['1d', '1wk', '1mo', '1y', 'max'],index=3)

    with col3:
        type_graphe = st.selectbox('Type de graphique', options=['Candlestick', 'Ligne'])

    with col4:
        indicateurs = st.multiselect('Indicateurs techniques', options=['MMS 20', 'MME 20'])

    # Mapping des intervalles
    intervalle_mapping = {
        '1d': '1m',
        '1wk': '30m',
        '1mo': '1d',
        '1y': '1wk',
        'max': '1wk'
    }
    if st.button('Mettre à jour'):
        intervalle = intervalle_mapping[periode]
        data = recuperer_donnees_boursieres(ticker, periode, intervalle)

        if not data.empty:
            data = traiter_donnees(data)
            data = ajouter_indicateurs_techniques(data)
            dernier_cours, variation, variation_pourcentage, haut, bas, volume = calculer_metriques(data)

            # Affichage des métriques principales
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label=f"{ticker} Dernier Prix", value=f"{dernier_cours:.2f} USD", delta=f"{variation:.2f} ({variation_pourcentage:.2f}%)")
            col2.metric("Haut", f"{haut:.2f} USD")
            col3.metric("Bas", f"{bas:.2f} USD")
            col4.metric("Volume", f"{volume:,}")

            # Graphique du cours actuel
            fig = go.Figure()
            if type_graphe == 'Candlestick':
                fig.add_trace(go.Candlestick(x=data['Datetime'],
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close']))
            else:
                fig = px.line(data, x='Datetime', y='Close')

            # Visualisation des indicateurs 
            for indicateur in indicateurs:
                if indicateur == 'MMS 20':
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['MMS_20'], name='MMS 20'))
                elif indicateur == 'MME 20':
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['MME_20'], name='MME 20'))

            # Format du graphique
            fig.update_layout(title=f'{ticker} {periode.upper()} Chart',
                              xaxis_title='Time',
                              yaxis_title='Price (USD)',
                              height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Affichage des données historiques et des indicateurs
            st.subheader('Données Historiques')
            st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

            st.subheader('Indicateurs Techniques')
            st.dataframe(data[['Datetime', 'MMS_20', 'MME_20']])
        else:
            st.warning("Impossible de récupérer les données pour le ticker sélectionné.")




def documentation():
    st.markdown("""
    <div style='text-align: center;'> <h1 style='color:#0E3453;'>Documentation</h1></div>
    """, unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:15px;color:#A75502'>Si vous souhaitez en savoir plus sur la conception de l'aplication, veuillez cliquer pour télécharger la documentation.</h6>",unsafe_allow_html=True)

    # Chargement du fichier PDF
    with open("media/Documentation_OptionLab.pdf", "rb") as file:
        pdf_data = file.read()

    # Encodage Base64
    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

    # Bouton de téléchargement centré
    st.markdown(
        f"""
        <style>
        .download-button {{
            display: flex;
            justify-content: center;
        }}
        .download-button a {{
            background-color: #0E3453; /* Bleu de l'image */
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }}
        .download-button a:hover {{
            background-color: #092436; /* Une teinte légèrement plus foncée pour le survol */
        }}
        </style>
        <div class="download-button">
            <a href="data:application/octet-stream;base64,{pdf_base64}" download="Documentation_OptionLab.pdf">
                Télécharger la documentation
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 =st.columns([1,4,1])
    # Dossier contenant les images
    image_folder = "media/docs"
    
    # Liste des fichiers dans le dossier, triée par nom
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        with col2:
            st.image(image, use_column_width=True)



# Associer les pages à leurs fonctions respectives
functions = {
    "accueil": accueil,
    "donnees": donnees,
    "pricing": pricing,
    "sensibilites": sensibilites,
    "visualisation": visualisation,
    "documentation": documentation,
}


# Afficher la page sélectionnée
current_page = get_current_page()
if current_page in functions:
    functions[current_page]()
else:
    st.write("Page non trouvée.")
