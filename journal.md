# Extraire les commentaires

## Méthode : 
API

## La structure de nontre corpus : 
En format csv avec quatre column:
- tag : recommende / not recommende
- commentaire
- date : date de commenter
- minute : minute totale de jouer

# Filtrer les commentaires
caractère non-sense
heure de jouer

# Preprocessing
stopword, lower, lemmatisation, ponctuation...

# Entrainement
1. annotation -> 1 doc = 1 classe (pas besion)
2. decoupage (dev / train / test)
3. vectorisation (countvect / tf-idf)
4. entrainement (train)
5. evaluation (test)