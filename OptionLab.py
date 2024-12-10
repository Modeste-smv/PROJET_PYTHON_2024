import streamlit as st
import sqlite3
import datetime
from importation import process_expirations


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
    st.write("Calcul du prix des options selon le modèle de Black-Scholes.")

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
