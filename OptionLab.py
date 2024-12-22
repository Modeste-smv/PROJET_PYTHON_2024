import numpy as np
import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from importation import process_expirations
import yfinance as yf


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

    # Connexion à la base de données
    conn = sqlite3.connect('options_data.db')
    cursor = conn.cursor()

    # Récupérer la liste des symboles uniques présents dans la base
    cursor.execute("SELECT DISTINCT Symbol FROM Ticker")
    symbols_in_db = [row[0] for row in cursor.fetchall()]

    conn.close()

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

        # On se connecte de nouveau pour récupérer les expirations correspondant au symbole choisi
        conn = sqlite3.connect('options_data.db')
        cursor = conn.cursor()
        # On suppose que la table Options contient toutes les dates d'expiration dont on a besoin
        cursor.execute("SELECT DISTINCT expiration_date FROM Options WHERE ticker = ?", (symbol,))
        all_expirations = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Convertir en datetime pour filtrer par année et mois
        df_exp = pd.DataFrame(all_expirations, columns=["expiration_date"])
        df_exp["expiration_date"] = pd.to_datetime(df_exp["expiration_date"]).dt.date

        # Extraire années disponibles
        df_exp['Year'] = df_exp['expiration_date'].apply(lambda d: d.year)
        df_exp['Month'] = df_exp['expiration_date'].apply(lambda d: d.month)

        years = sorted(df_exp['Year'].unique())
        selected_year = st.selectbox("Année d'expiration", options=years)

        # Filtrer par année
        df_year = df_exp[df_exp['Year'] == selected_year]
        months = sorted(df_year['Month'].unique())
        selected_month = st.selectbox("Mois d'expiration", options=months, format_func=lambda m: f"{m:02d}")

        # Filtrer par mois
        df_month = df_year[df_year['Month'] == selected_month]

        # Extraire les jours disponibles
        available_dates = sorted(df_month['expiration_date'].unique())
        expiration_date = st.selectbox("Date d'expiration", options=available_dates)

        # Saisie du strike price (prix d'exercice)
        conn = sqlite3.connect('options_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT strike FROM Options
            WHERE ticker = ? AND optionType = ? AND expiration_date = ?
            ORDER BY strike
        """, (symbol, option_type, expiration_date.strftime("%Y-%m-%d")))
        all_strikes = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not all_strikes:
            st.warning("Aucun prix d'exercice disponible pour cette date d'expiration.")
            return

        # Si les strikes sont nombreux et réguliers, on peut utiliser un slider
        # Sinon, un selectbox est plus approprié
        # Exemple avec un selectbox (plus sûr si irrégulier) :
        strike_price = st.selectbox("Prix d'exercice (Strike price)", options=all_strikes)


        # Bouton pour calculer la valeur théorique
        if st.button("Calculer la valeur de l'option"):
            # Ici, on mettra plus tard le code qui :
            # 1. Récupère le prix du sous-jacent
            # 2. Récupère la volatilité, le taux sans risque, etc.
            # 3. Calcule la valeur théorique avec la formule de Black-Scholes
            # 4. Affiche le résultat
            
            st.write("La fonctionnalité de calcul est à venir...")
    else:
        # Si c'est un acheteur, on traitera plus tard
        st.write("La fonctionnalité pour les acheteurs sera implémentée ultérieurement.")

def sensibilites():
    st.title('📊 Sensibilités')
    st.write("Analyse des sensibilités (Greeks) des options.")

    # Connexion à la base de données
    conn = sqlite3.connect("options_data.db")  # Remplacez par le chemin de votre base SQLite

    # Chargement des données depuis la base de données
    query = "SELECT * FROM options"
    data = pd.read_sql(query, conn)

    # Filtrer les données selon l'utilisateur
    tickers = data['ticker'].unique()
    selected_ticker = st.selectbox("Sélectionnez un ticker", tickers)
    filtered_data = data[data['ticker'] == selected_ticker]

    expirations = filtered_data['expiration_date'].unique()
    selected_expiration = st.selectbox("Sélectionnez une date d'expiration", expirations)
    filtered_data = filtered_data[filtered_data['expiration_date'] == selected_expiration]

    strikes = filtered_data['strike'].unique()
    selected_strike = st.selectbox("Sélectionnez un prix d'exercice (strike)", strikes)
    filtered_data = filtered_data[filtered_data['strike'] == selected_strike]

    option_type = st.radio("Type d'option", ['call', 'put'])
    option_data = filtered_data[filtered_data['optionType'] == ('C' if option_type == 'call' else 'P')]

    # Vérifier si des données sont disponibles
    if option_data.empty:
        st.error("Aucune donnée correspondante trouvée. Veuillez ajuster vos sélections.")
        return

    # Vérifier la colonne 'impliedVolatility'
    if 'impliedVolatility' not in option_data.columns or option_data['impliedVolatility'].isnull().all():
        st.error("La colonne 'impliedVolatility' est vide ou absente. Vérifiez les données.")
        return

    ticker = yf.Ticker(selected_ticker)
    S0 = ticker.history(period="1d")['Close'].iloc[-1]  # Prix actuel du sous-jacent
    T = (datetime.strptime(selected_expiration, '%Y-%m-%d').date() - datetime.now().date()).days / 365.0
    K = selected_strike
    sigma = option_data['impliedVolatility'].iloc[0]
    r = st.number_input("Taux sans risque (r, en %)", value=5.0) / 100
    N = st.number_input("Nombre de trajectoires Monte Carlo (N)", value=100000, step=1000)
    M = st.number_input("Nombre de pas dans la simulation (M)", value=100, step=10)

    def calcul_sensibilites(S0, K, T, r, sigma, N, M, option_type="call"):
        dt = T / M
        discount = np.exp(-r * dt)

        # Simuler les trajectoires de prix du sous-jacent
        S = np.zeros((N, M + 1))
        S[:, 0] = S0
        for t in range(1, M + 1):
            Z = np.random.standard_normal(N)
            S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        # Calcul des payoffs à l'échéance
        if option_type == "call":
            payoff = np.maximum(S[:, -1] - K, 0)
        else:
            payoff = np.maximum(K - S[:, -1], 0)

        # Estimation du prix de l'option
        price = discount * np.mean(payoff)

        # Calcul des Greeks par différences finies
        h = 0.01

        # Delta
        payoff_up = np.maximum((S0 + h) * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(N)) - K, 0)
        payoff_down = np.maximum((S0 - h) * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(N)) - K, 0)
        price_up = discount * np.mean(payoff_up)
        price_down = discount * np.mean(payoff_down)
        delta = (price_up - price_down) / (2 * h)

        # Gamma
        gamma = (price_up - 2 * price + price_down) / (h ** 2)

        # Vega
        payoff_vega = np.maximum(S0 * np.exp((r - 0.5 * (sigma + h)**2) * T + (sigma + h) * np.sqrt(T) * np.random.standard_normal(N)) - K, 0)
        price_vega = discount * np.mean(payoff_vega)
        vega = (price_vega - price) / h

        # Theta
        payoff_theta = np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * (T - h) + sigma * np.sqrt(T - h) * np.random.standard_normal(N)) - K, 0)
        price_theta = discount * np.mean(payoff_theta)
        theta = (price_theta - price) / h

        # Rho
        payoff_rho = np.maximum(S0 * np.exp((r + h - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(N)) - K, 0)
        price_rho = np.mean(payoff_rho) * np.exp(-(r + h) * T)
        rho = (price_rho - price) / h

        return price, delta, gamma, vega, theta, rho

    if st.button("Calculer"):
        price, delta, gamma, vega, theta, rho = calcul_sensibilites(S0, K, T, r, sigma, int(N), int(M), option_type)
        st.write(f"### Résultats :")
        st.write(f"- **Prix de l'option** : {price:.4f}")
        st.write(f"- **Delta** : {delta:.4f}")
        st.write(f"- **Gamma** : {gamma:.4f}")
        st.write(f"- **Vega** : {vega:.4f}")
        st.write(f"- **Theta** : {theta:.4f}")
        st.write(f"- **Rho** : {rho:.4f}")

    # Fermeture de la connexion à la base de données
    conn.close()


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