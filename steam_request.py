import tkinter as tk
from tkinter import ttk
import webbrowser
import requests
from bs4 import BeautifulSoup
import re

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
from extract_reviews import get_steam_reviews, save_reviews_to_csv

app_id = app_id_global 
reviews = get_steam_reviews(app_id=app_id, num_reviews=1000)
save_reviews_to_csv(reviews, filename=f'data/{game_name}.csv'.replace(" ", "_"))
