import joblib

# 1. Charger le modèle et le vectoriseur
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Nouveaux commentaires à tester
test_comments = [
    # Positifs
    "Un chef-d’œuvre, j’y joue depuis des semaines sans m’en lasser !",
    "L’ambiance est incroyable, et la bande-son me donne des frissons.",
    "Gameplay fluide, graphismes superbes, je recommande à 100%.",
    "Une vraie pépite, surtout pour les fans du genre roguelike.",
    "Solo comme multi sont excellents, une très bonne surprise.",
    "Chaque mise à jour apporte du contenu de qualité, bravo aux devs.",
    "L’histoire est prenante du début à la fin, très bon doublage aussi.",
    "Très bien optimisé, tourne nickel même sur une config moyenne.",
    "J’ai adoré la personnalisation du perso et la liberté de choix.",
    "Un incontournable pour les fans de jeux narratifs.",

    # Neutres
    "Bon jeu dans l’ensemble, mais un peu court pour le prix.",
    "Pas mal, mais il manque encore un mode coop local.",
    "J’attends encore des patchs pour corriger les petits bugs.",
    "Graphiquement pas fou, mais le gameplay rattrape le tout.",
    "L’idée est bonne, mais l’exécution est un peu bancale.",

    # Négatifs
    "Très déçu, le jeu plante au bout de 10 minutes.",
    "Système de progression frustrant, on grind trop pour avancer.",
    "Pay-to-win à mort, injouable sans passer par la boutique.",
    "Contrôles imprécis, caméra horrible dans les combats.",
    "Serveurs constamment en panne, expérience multijoueur gâchée."
]

X_test = vectorizer.transform(test_comments)

predictions = model.predict(X_test)


for commentaire, prediction in zip(test_comments, predictions):
    label = "RECOMMANDÉ" if prediction == 1 else "NON RECOMMANDÉ"
    print(f"{label} --> {commentaire}")
