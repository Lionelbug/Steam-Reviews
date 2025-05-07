import tkinter as tk
from tkinter import ttk
import webbrowser
import requests
from bs4 import BeautifulSoup
import re
import csv
import os
from datetime import datetime
import pandas as pd

################################
#########Random Forest##########
################################
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

""" .exe file in progress
pyinstaller --onefile --collect-all imblearn --add-data "data/Stopwords.txt:data" --add-data "features:features" --add-data "~/Extraction d'info/testsp/lib64/python3.12/site-packages/imblearn/VERSION.txt:imblearn" steam_request.py
dossier = "./data/files_for_corpus/"
nom_sortie = "./data/corpus.csv"
os.makedirs(dossier, exist_ok=True)
os.makedirs(os.path.dirname(nom_sortie), exist_ok=True)
"""

################################
#########Steam Request##########
################################

app_id_global = None
game_name = None

def search_steam():
    query = entry.get()
    if not query:
        return

    url = f"https://store.steampowered.com/search/?term={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        result_label.config(text="Erreur lors de la récupération des données.")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for game in soup.select(".search_result_row")[:5]:
        title = game.select_one(".title").text
        link = game["href"]
        results.append((title, link))

    display_results(results)

def display_results(results):
    for widget in result_frame.winfo_children():
        widget.destroy()

    for title, link in results:
        frame = tk.Frame(result_frame)
        frame.pack(fill="x", padx=5, pady=2)

        label = tk.Label(frame, text=title, anchor="w")
        label.pack(side="left", fill="x", expand=True)

        btn = ttk.Button(frame, text="Ouvrir", command=lambda url=link: webbrowser.open(url))
        btn.pack(side="right")

        btn2 = ttk.Button(frame, text="Analyse", command=lambda url=link, t=title: analyze_link(url, t))
        btn2.pack(side="left")

def analyze_link(link, title):
    global app_id_global
    global game_name

    app_id = extract_steam_app_id(link)
    app_id_global = app_id
    game_name = title
    id_label.config(text=f"ID de {title} : {app_id}")

    reviews = get_steam_reviews(app_id=app_id, num_reviews=10000)
    cleaned_name = re.sub(r'[^a-zA-Z0-9]', '_', game_name)
    #cleaned_name = cleaned_name.replace('__', '_')
    filename = f'./data/files_for_corpus/{cleaned_name}.csv'
    save_reviews_to_csv(reviews, filename)
    id_label.config(text=f"{title} (ID: {app_id}) - Avis sauvegardés dans {filename}")

def extract_steam_app_id(url):
    match = re.search(r'/app/(\d+)', url)
    if match:
        return match.group(1)
    return "No ID"

def get_steam_reviews(app_id, num_reviews=10000):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        'json': 1,
        'filter': 'recent',
        'language': 'french',
        'day_range': 365,
        'review_type': 'all',
        'purchase_type': 'steam',
        'num_per_page': 100,
        'cursor': '*',
    }

    all_reviews = []
    count = 0

    while count < num_reviews:
        response = requests.get(url, params=params)
        data = response.json()
        reviews = data.get('reviews', [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        count += len(reviews)
        params['cursor'] = data['cursor']

    return all_reviews[:num_reviews]

def save_reviews_to_csv(reviews, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tag', 'comment', 'date', 'minutes'])

        for review in reviews:
            voted = 'recommande' if review['voted_up'] else 'non_recommande'
            content = review['review'].replace('\n', ' ').strip()
            time_str = datetime.fromtimestamp(review['timestamp_created']).strftime('%Y-%m-%d %H:%M:%S')
            playtime = review['author'].get('playtime_at_review', 0)

            writer.writerow([voted, content, time_str, playtime])

################################
#########  Interface  ##########
################################

root = tk.Tk()
root.title("Recherche Steam")
root.geometry("500x500")
#icon = tk.PhotoImage(file="./features/lens.png")
#root.iconphoto(False, icon)
frame_top = tk.Frame(root)
frame_top.pack(pady=10)

entry = tk.Entry(frame_top, width=40)
entry.pack(side="left", padx=5)

btn_search = ttk.Button(frame_top, text="Rechercher", command=search_steam)
btn_search.pack(side="left")

result_label = tk.Label(root, text="Résultats :", font=("Arial", 15, "bold"))
result_label.pack(pady=5)

result_frame = tk.Frame(root)
result_frame.pack(fill="both", expand=True)

id_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
id_label.pack(pady=10)

root.mainloop()


################################
#########   Fusion    ##########
################################


dossier = "./data/files_for_corpus/"
nom_sortie = "./data/corpus.csv"

# === Récupération de tous les fichiers CSV ===
fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

# === Fusion ===
with open(nom_sortie, mode='w', newline='', encoding='utf-8') as sortie:
    writer = None
    for index, nom_fichier in enumerate(fichiers_csv):
        chemin_fichier = os.path.join(dossier, nom_fichier)
        print(f"Traitement du fichier {nom_fichier}...")  # Afficher quel fichier est en cours de traitement
        try:
            with open(chemin_fichier, mode='r', encoding='utf-8') as f:
                lecteur = csv.reader(f)
                try:
                    en_tete = next(lecteur)  # Lire l'en-tête une seule fois
                except StopIteration:
                    print(f"Le fichier {nom_fichier} est vide, il sera ignoré.")
                    continue  # Passer au fichier suivant si celui-ci est vide

                if writer is None:
                    writer = csv.writer(sortie)
                    writer.writerow(en_tete)  # Écrire l’en-tête une fois

                ligne_count = 0  # Compteur de lignes traitées
                for ligne in lecteur:
                    writer.writerow(ligne)
                    ligne_count += 1

                print(f"{ligne_count} lignes ont été ajoutées depuis {nom_fichier}.")  # Affichage du nombre de lignes traitées

        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {nom_fichier}: {e}")
            continue  # Passer au fichier suivant en cas d'erreur

print(f"{len(fichiers_csv)} fichiers fusionnés avec succès dans le corpus.")

corpus_path = './data/corpus.csv'
stopwords_path = './data/Stopwords.txt'

# Charger le corpus CSV
corpus_df = pd.read_csv(corpus_path)

# Charger les stopwords
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

# Nettoyage du texte : mettre en minuscule et supprimer les stopwords
def clean_text(text):
    text = str(text)
    # Mettre en minuscule
    text = text.lower()
    # Diviser le texte en mots et supprimer les stopwords
    cleaned_text = ' '.join([word for word in text.split() if word not in stopwords])
    return cleaned_text

# Appliquer le nettoyage à chaque commentaire dans le corpus
corpus_df['comment'] = corpus_df['comment'].apply(clean_text)

# Sauvegarder le corpus nettoyé en écrasant le fichier original
corpus_df.to_csv(corpus_path, index=False)

# Afficher les premières lignes du corpus nettoyé pour vérifier
print(corpus_df.head())

df = pd.read_csv("./data/corpus.csv")

# 2. Nettoyer et préparer les données
df['label'] = df['tag'].map({'recommande': 1, 'non_recommande': 0})
df = df.dropna(subset=['comment', 'label'])  # Supprimer les lignes incomplètes

print("Répartition des labels avant SMOTE:")
print(df['label'].value_counts())

df = df.dropna(subset=['comment', 'label'])  # Supprimer les lignes incomplètes

# 3. Variables explicatives et cible
X = df['comment']
y = df['label']

# 4. Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Appliquer SMOTE pour équilibrer les classes dans l'ensemble d'entraînement
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# 5. Pipeline avec Random Forest
pipeline = make_pipeline(
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# 6. Entraîner le modèle avec les données rééchantillonnées
pipeline.fit(X_train_resampled, y_train_resampled)

# 7. Évaluer le modèle
y_pred = pipeline.predict(X_test_tfidf)
print("Répartition des labels après SMOTE sur l'ensemble d'entraînement:")
print(pd.Series(y_train_resampled).value_counts())
print(classification_report(y_test, y_pred))

# Test sur de nouveaux commentaires
test_comments = [
    "Une pépite ! L’ambiance et la bande-son sont juste parfaites.",
    "Gameplay ultra-dynamique et scénario prenant — une vraie réussite.",
    "Je passe des heures dessus sans voir le temps passer, quelle claque !",
    "Un chef-d’œuvre narratif avec des personnages inoubliables.",
    "Le meilleur jeu de rôle depuis des années, bravo aux développeurs !",
    "Les graphismes sont magnifiques, mais l’IA des ennemis est catastrophique.",
    "Trop répétitif, on fait la même mission en boucle avec peu de variété.",
    "Les microtransactions gâchent complètement l’expérience.",
    "Des problèmes de lag constants, même avec une connexion fibre…",
    "L’interface est mal foutue et rend le jeu frustrant à jouer.",
    "Jeu sorti trop tôt : bugs, crashes… À éviter en l’état.",
    "Un bon potentiel, mais le multijoueur est déséquilibré et plein de cheaters.",
    "La campagne est courte et sans grand intérêt, dommage.",
    "Les développeurs écoutent la communauté et améliorent le jeu régulièrement — chapeau !",
    "Un must-have pour les fans du genre, malgré quelques défauts mineurs."
]

test_comments_tfidf = vectorizer.transform(test_comments)  # Appliquer la même transformation TF-IDF sur les commentaires de test

preds = pipeline.predict(test_comments_tfidf)

for comment, pred in zip(test_comments, preds):
    print(f"{'RECOMMENDED' if pred == 1 else 'NOT RECOMMENDED'} --> {comment}")


################################
#########    Model    ##########
################################
#joblib.dump(pipeline, 'random_forest_model.pkl')
#joblib.dump(vectorizer, 'vectorizer.pkl')
