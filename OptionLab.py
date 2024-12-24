import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from importation import process_expirations
import s3fs
import yfinance as yf
import matplotlib.pyplot as plt


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
    st.title('Données')
    st.write("Veuillez saisir les symboles (séparés par des virgules). Exemple : AAPL, MSFT, GOOGL")

    # Champ pour saisir les symboles
    symbol_input = st.text_input("Symboles", value="AAPL")
    symbols = [sym.strip() for sym in symbol_input.split(",") if sym.strip()]

    st.write("Optionnel : Sélectionnez une plage de dates (min et max) pour filtrer les dates d'expiration disponibles.")
    # Date minimale
    min_date = st.date_input("Date minimale", value=None)
    # Date maximale
    max_date = st.date_input("Date maximale", value=None)

    # Bouton pour lancer l'importation
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

            # Connexion à la base de données
            conn = sqlite3.connect('options_data.db')
            cursor = conn.cursor()

            # Vider les tables Ticker et Options
            cursor.execute("DELETE FROM Options")
            cursor.execute("DELETE FROM Ticker")
            conn.commit()

            # Insérer les nouveaux symboles
            unique_symbols = data['ticker'].unique()
            for sym in unique_symbols:
                cursor.execute("INSERT INTO Ticker (Symbol) VALUES (?)", (sym,))
            conn.commit()

            # Insérer les données dans Options
            data.to_sql('Options', conn, if_exists='append', index=False)

            st.success("Les données ont été insérées dans la base de données (base vidée avant insertion).")
            conn.close()

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
    
    elif role == "Acheteur":
        # Partie acheteur
        if symbols_in_db:
            symbol = st.selectbox("Choisissez un symbole d'actif", options=symbols_in_db)
        else:
            st.warning("Aucun symbole n'est disponible dans la base de données.")
            return

        # Choix du type d'option
        option_type = st.selectbox("Type d'option :", options=["Call", "Put"])

        # Filtrer les données pour récupérer les expirations disponibles
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
        available_dates = sorted(df_month['expiration_date'].unique())
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
                    r = 0.05  # Taux sans risque (exemple)
                    sigma = row['impliedVolatility']
                    M = 50  # Nombre de pas
                    simulations = 10000

                    american_option = AmericanOptionsLSMC(option_type.lower(), S0, K, T, M, r, 0, sigma, simulations)
                    theoretical_values.append(american_option.price())
                    last_prices.append(row['lastPrice'])

                # Création du graphique
                fig, ax = plt.subplots()

                # Tracer les courbes des valeurs théoriques et des derniers prix
                ax.plot(strikes, theoretical_values, label="Valeur théorique", color="turquoise")
                ax.plot(strikes, last_prices, label="Valeur de marché", color="coral")

                # Configuration des axes et légendes
                ax.set_xlabel("Prix d'exercice (Strike)", fontsize=8)
                ax.set_ylabel("Prix", fontsize=8)
                ax.set_title(f"{symbol} ({option_type}) - {expiration_date.date()}", fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.legend(fontsize=6)

                # Affichage du graphique
                st.pyplot(fig)

                # Calcul de l'option à conseiller
                differences = [(tv - lp) for tv, lp in zip(theoretical_values, last_prices)]
                max_diff_index = differences.index(max(differences))
                best_strike = strikes[max_diff_index]
                best_theoretical_value = theoretical_values[max_diff_index]
                best_last_price = last_prices[max_diff_index]

                # Message de recommandation
                st.markdown(f"### Conseil :")
                st.markdown(f"L'option avec le prix d'exercice (strike) de **{best_strike}** est recommandée.")
                st.markdown(f"Elle a la plus grande différence entre la valeur théorique (**{best_theoretical_value:.2f}**) et le dernier prix (**{best_last_price:.2f}**).")

    else:
        st.warning("Aucune option disponible pour cette date d'expiration.")

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




def visualisation():
    st.title('🔍 Visualisation')
    st.write("Visualisations graphiques des données et des résultats.")

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
