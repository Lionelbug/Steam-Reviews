import requests
import time
import csv
from datetime import datetime

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
app_id = '1245620'  # Elden Ring
reviews = get_steam_reviews(app_id=app_id, num_reviews=200)
save_reviews_to_csv(reviews, filename='elden_ring_reviews.csv')