import tkinter as tk
from tkinter import ttk
import webbrowser
import requests
from bs4 import BeautifulSoup
import re
import time
import csv
from datetime import datetime


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

    for game in soup.select(".search_result_row")[:5]:  #
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
    print(f"Le lien est le suivant : {link}")
    app_id = extract_steam_app_id(link)
    print(f"L'ID de {title} est : {app_id}")
    app_id_global = app_id
    game_name = title
    root.destroy()


def extract_steam_app_id(url):
    match = re.search(r'/app/(\d+)', url)
    if match:
        return match.group(1)
    return "No ID"

# ===Interface Tkinter===
#
root = tk.Tk()
root.title("Recherche Steam")
root.geometry("500x400")

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

id_label = tk.Label(root, text="", font=("Arial", 12))
id_label.pack(pady=10)


root.mainloop()

print(app_id_global) # Récupère le texte du Label

#===Extract reviews===
"""import requests
import time
import csv
from datetime import datetime"""

# https://partner.steamgames.com/doc/store/getreviews?

def get_steam_reviews(app_id, num_reviews=100): # récupérer les commentaires d'un jeu
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        'json': 1,
        'filter': 'recent',
        'language': 'english',  # désigner la langue
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
        time.sleep(1)  # attendre 1 seconde

    return all_reviews[:num_reviews]

def save_reviews_to_csv(reviews, filename='steam_reviews.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tag', 'comment', 'date', 'minutes'])

        for review in reviews:
            voted = 'recommende' if review['voted_up'] else 'not recommende'
            content = review['review'].replace('\n', ' ').strip()
            time_str = datetime.fromtimestamp(review['timestamp_created']).strftime('%Y-%m-%d %H:%M:%S')
            playtime = review['author'].get('playtime_at_review', 0)

            writer.writerow([voted, content, time_str, playtime])

# un exemple
app_id = app_id_global # Elden Ring
reviews = get_steam_reviews(app_id=app_id, num_reviews=200)
save_reviews_to_csv(reviews, filename=f'{game_name}.csv'.replace(" ", "_"))
