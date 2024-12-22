import streamlit as st
import sqlite3
import datetime
import pandas as pd
from importation import process_expirations
import s3fs
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'})
data = pd.read_parquet("s3://modestesmv/database.parquet", filesystem=fs)

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
