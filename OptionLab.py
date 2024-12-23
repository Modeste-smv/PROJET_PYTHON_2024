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




# Liaison à la base
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'})
data = pd.read_parquet("s3://modestesmv/database.parquet", filesystem=fs)
# Conversion explicite de la colonne expiration_date en datetime
data['expiration_date'] = pd.to_datetime(data['expiration_date'], errors='coerce')

# SIMULATION MONTE CARLO 
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





class AmericanOptionsLSMC:
    """Classe pour valoriser les options américaines en utilisant la méthode de Longstaff-Schwartz (2001)."""

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

    def _generate_price_paths(self, seed=123):
        np.random.seed(seed)
        prices = np.zeros((self.M + 1, self.simulations))
        prices[0, :] = self.S0

        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = np.concatenate((brownian, -brownian))
            prices[t, :] = prices[t - 1, :] * np.exp(
                (self.r - self.sigma ** 2 / 2) * self.time_unit + self.sigma * np.sqrt(self.time_unit) * brownian
            )
        return prices

    def _calculate_payoffs(self, prices):
        if self.option_type == 'call':
            return np.maximum(prices - self.strike, 0)
        elif self.option_type == 'put':
            return np.maximum(self.strike - prices, 0)

    def _backward_induction(self, prices, payoffs):
        values = np.zeros_like(payoffs)
        values[-1, :] = payoffs[-1, :]

        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(prices[t, :], values[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, prices[t, :])
            values[t, :] = np.where(payoffs[t, :] > continuation_value, payoffs[t, :], values[t + 1, :] * self.discount)

        return values[1, :] * self.discount

    def price(self):
        prices = self._generate_price_paths()
        payoffs = self._calculate_payoffs(prices)
        values = self._backward_induction(prices, payoffs)
        return np.mean(values)





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
    # Style for the header
    header_props = [('background-color', '#0E3453'),  # Blue background
                    ('color', 'white'),               # White text
                    ('border', '1px solid black'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')]

    # Style for the body
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
# Fonction pour obtenir la page actuelle
def get_current_page():
    return st.session_state.current_page

# Définition des fonctions pour chaque page
def accueil():
    st.title('🏠 Accueil')
    st.write("Bienvenue dans l'application de pricing des options !")


def donnees():
    st.markdown("""
    <div style='text-align: center;'> <h1 style='color:#0E3453;'>Données</h1></div>
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







def pricing():
    st.title('📈 Pricing')
    st.write("Calculez la valeur théorique de votre option.")

    # Vérifier si la base de données est chargée
    if data.empty:
        st.warning("La base de données est vide. Veuillez vérifier la source.")
        return

    # Récupérer les symboles uniques disponibles
    symbols_in_db = data['ticker'].unique().tolist()

    # Choix du rôle : Acheteur ou Vendeur
    role = st.radio("Êtes-vous un acheteur ou un vendeur ?", options=["Acheteur", "Vendeur"])

    if role == "Vendeur":
        # Si c'est un vendeur, on demande les paramètres nécessaires
        
        # Saisie du symbole
        if symbols_in_db:
            symbol = st.selectbox("Symbole de l'actif", options=symbols_in_db)
        else:
            st.warning("Aucun symbole n'est disponible dans la base de données.")
            return
            
        # Choix du type d'option
        option_type = st.selectbox("Type d'option :", options=["Call", "Put"])

        # Filtrer les données pour récupérer les expirations disponibles
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
        available_dates = sorted(df_month['expiration_date'].unique())
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
        # Récupérer les données pour AAPL
        ticker = yf.Ticker(symbol)
        # Méthode 1 : Récupérer le prix actuel directement
        S0 = ticker.history(period="1d")['Close'].iloc[-1]
        K = selected_row['strike']
        T = (pd.to_datetime(selected_row['expiration_date'], unit='ms') - pd.Timestamp.now()).days / 365.0
        r = 0.05  # Exemple de taux sans risque
        sigma = selected_row['impliedVolatility']
        M = 50  # Nombre de pas
        simulations = 10000

        american_option = AmericanOptionsLSMC(option_type.lower(), S0, K, T, M, r, 0, sigma, simulations)
        option_price = american_option.price()

        st.success(f"Valeur théorique de l'option américaine : {option_price:.2f} €")
        st.write(selected_row['lastPrice'])
    else:
        # Si c'est un acheteur, on traitera plus tard
        st.write("La fonctionnalité pour les acheteurs sera implémentée ultérieurement.")

##################################### SENSIBILITE ##############################################
def sensibilites():
    st.title('📊 Sensibilités')
    st.write("Analyse des sensibilités (Greeks) des options.")

    # Filtrer les données selon les sélections de l'utilisateur
    tickers = data['ticker'].unique()
    selected_ticker = st.selectbox("Sélectionnez un ticker", tickers)
    filtered_data = data[data['ticker'] == selected_ticker]

    # Sélection de l'année
    filtered_data['Year'] = filtered_data['expiration_date'].dt.year
    years = filtered_data['Year'].unique()
    selected_year = st.selectbox("Sélectionnez une année d'expiration", sorted(years))

    # Filtrer les données par année sélectionnée
    filtered_year_data = filtered_data[filtered_data['Year'] == selected_year]

    # Sélection du mois
    filtered_year_data['Month'] = filtered_year_data['expiration_date'].dt.month
    months = filtered_year_data['Month'].unique()
    selected_month = st.selectbox("Sélectionnez un mois d'expiration", sorted(months))

    # Filtrer les données par mois sélectionné
    filtered_month_data = filtered_year_data[filtered_year_data['Month'] == selected_month]

    # Sélection de la date précise
    available_dates = filtered_month_data['expiration_date'].dt.date.unique()
    selected_date = st.selectbox("Sélectionnez une date d'expiration", sorted(available_dates))

    # Affiner les données pour la date sélectionnée
    filtered_data = filtered_month_data[filtered_month_data['expiration_date'].dt.date == selected_date]

    # Sélection du strike
    strikes = filtered_data['strike'].unique()
    selected_strike = st.selectbox("Sélectionnez un prix d'exercice (strike)", strikes)

    # Type d'option
    option_type = st.radio("Type d'option", ['Call', 'Put'])

    # Filtrer les données finales
    option_data = filtered_data[
        (filtered_data['strike'] == selected_strike) &
        (filtered_data['optionType'] == option_type)
    ]

    # Vérifier si des données sont disponibles
    if option_data.empty:
        st.error("Aucune donnée correspondante trouvée. Veuillez ajuster vos sélections.")
        return

    # Vérifier la colonne 'impliedVolatility'
    if 'impliedVolatility' not in option_data.columns or option_data['impliedVolatility'].isnull().all():
        st.error("La colonne 'impliedVolatility' est vide ou absente. Vérifiez les données.")
        return

    # Identifier la ligne exacte
    selected_row = option_data.iloc[0]

    # Récupération des paramètres à partir de la ligne sélectionnée
    ticker = yf.Ticker(selected_ticker)
    try:
        S0 = ticker.history(period="1d")['Close'].iloc[-1]  # Prix actuel du sous-jacent
    except Exception as e:
        st.error(f"Impossible de récupérer le prix actuel : {e}")
        return

    T = (pd.to_datetime(selected_date) - pd.Timestamp.now()).days / 365.0  # Maturité
    K = selected_row['strike']
    sigma = selected_row['impliedVolatility']
    r = st.number_input("Taux sans risque (r, en %)", value=5.0) / 100
    N = st.number_input("Nombre de trajectoires Monte Carlo (N)", value=100000, step=1000)
    M = st.number_input("Nombre de pas dans la simulation (M)", value=100, step=10)

    # Fonction pour calculer les sensibilités
    def calcul_sensibilites(S0, K, T, r, sigma, N, M, option_type="call"):
        dt = T / M
        discount = np.exp(-r * dt)

        # Simuler les trajectoires
        S = np.zeros((N, M + 1))
        S[:, 0] = S0
        for t in range(1, M + 1):
            Z = np.random.standard_normal(N)
            S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        # Payoff
        if option_type == "call":
            payoff = np.maximum(S[:, -1] - K, 0)
        else:
            payoff = np.maximum(K - S[:, -1], 0)

        # Prix
        price = discount * np.mean(payoff)

        # Greeks
        h = 0.01

        # Delta
        S_up = S0 + h
        S_down = S0 - h
        payoff_up = np.maximum(S_up - K, 0) if option_type == "call" else np.maximum(K - S_up, 0)
        payoff_down = np.maximum(S_down - K, 0) if option_type == "call" else np.maximum(K - S_down, 0)
        delta = (np.mean(payoff_up) - np.mean(payoff_down)) / (2 * h)

        # Gamma
        gamma = (np.mean(payoff_up) - 2 * price + np.mean(payoff_down)) / (h ** 2)

        # Vega
        sigma_up = sigma + h
        payoff_vega = np.maximum(S0 * np.exp((r - 0.5 * sigma_up**2) * T + sigma_up * np.sqrt(T) * np.random.standard_normal(N)) - K, 0)
        vega = (np.mean(payoff_vega) - price) / h

        # Theta
        T_down = T - h
        payoff_theta = np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T_down + sigma * np.sqrt(T_down) * np.random.standard_normal(N)) - K, 0)
        theta = (np.mean(payoff_theta) - price) / h

        return price, delta, gamma, vega, theta

    # Calculer et afficher les résultats
    if st.button("Calculer"):
        price, delta, gamma, vega, theta = calcul_sensibilites(S0, K, T, r, sigma, int(N), int(M), option_type)
        st.write("### Résultats :")
        st.write(f"- **Prix de l'option** : {price:.4f}")
        st.write(f"- **Delta** : {delta:.4f}")
        st.write(f"- **Gamma** : {gamma:.4f}")
        st.write(f"- **Vega** : {vega:.4f}")
        st.write(f"- **Theta** : {theta:.4f}")



############################################# ACTIONS EN TEMPS RÉEL #############################################

##########################################################################################
## PART 1 : Fonctions pour Récupérer, Traiter et Créer des Indicateurs Techniques##
##########################################################################################

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

# Traiter les données boursières

def traiter_donnees(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Calculer les métriques principales

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

##############################################################################################
#################### PART 2: Création de la Mise en Page de Visualisation ###################
##############################################################################################

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

            # Afficher les métriques principales

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

            # visualisation des indicateurs 
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

            # Display historical data and technical indicators
            st.subheader('Données Historiques')
            st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

            st.subheader('Indicateurs Techniques')
            st.dataframe(data[['Datetime', 'MMS_20', 'MME_20']])
        else:
            st.warning("Impossible de récupérer les données pour le ticker sélectionné.")





def comparaison():
    st.title('⚖️ Comparaison')
    st.write("Comparaison des modèles de pricing des options.")

def aide():
    st.title('❓ Aide')
    st.write("Documentation et assistance pour l'utilisation de l'application.")

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
