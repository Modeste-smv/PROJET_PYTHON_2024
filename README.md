# PROJET_PYTHON_2024
Membres : Modeste, Enzo &amp; Lina

# Guide utilisateur pour lancer notre application OptionLab sur Streamlit

## Étape 1 : Lancer le fichier contenant toutes les données nécessaires
Ouvrez un nouveau terminal et utilisez la commande suivante : 

```
cd PROJET_PYTHON_2024
```

## Étape 2 : Préparer votre environnement
Le fichier `requirements.txt` contient tous les prérequis nécessaires au bon fonctionnement du programme. Pour s'assurer que ceux-ci sont bien installés, utilisez la commande suivante (dans le terminal)

```
python -m pip install -r requirements.txt
```

## Étape 3 : Naviguer vers le dossier contenant l'application
Placez-vous dans le répertoire `OptionLab.py` où se trouve le fichier de l'application Streamlit. 
Vous pourrez voir le détail des calculs et les méthodes utilisées pour nos analyses financières.

## Étape 4 : Obtenir l'accès à la base de données
Exécutez la commande suivante dans le terminal :

```
export AWS_ACCESS_KEY_ID=SA-PYTHON
export AWS_SECRET_ACCESS_KEY=UILHK?LU89UM0.OKK.J?UO651
export AWS_SESSION_TOKEN=
```

## Étape 5 : Lancer l'application Streamlit
Exécutez la commande suivante dans le terminal :

```
streamlit run OptionLab.py
```

## Étape 6 : Accéder à l'application
Une fois la commande exécutée, vous verrez un message contenant un lien généré dans le terminal, similaire à ceci :

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8510
Network URL: http://10.233.115.215:8510
External URL: http://159.180.239.166:8510
```

Cliquez sur le lien `http://localhost:8501` ou copiez-le dans votre navigateur pour ouvrir l'application.
Vous pouvez désormais naviguer librement sur notre application OptionLab.
En cas de difficultés, ou si vous voulez en savoir plus sur le développement de notre application, référez-vous à notre documentation qui se trouve en format pdf dans l'onglet "Documentation".
Nous espérons qu'OptionLab vous fournira de précieux conseils dans vos futures décisions d'investissement.