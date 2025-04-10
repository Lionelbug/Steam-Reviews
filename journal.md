# Extraire les commentaires sur Steam

## Méthode : 
Pour récupérer les commentaires des utilisateurs, nous utilisons l’API mise à disposition par Steam. 
La documentation officielle est disponible à cette adresse : https://partner.steamgames.com/doc/store/getreviews

## La structure de nontre corpus : 
Les données extraites sont sauvegardées dans des fichiers CSV, chacun contenant les sept colonnes suivantes :
- tag : recommende / not recommende
- helpful : nombre de joueur qui trouve ce commentaire utile
- weighted_vote_score : la note d'importance
- commentaire
- date : date de commenter
- minutes : minutes totales de jouer
- early_access : si le joueur commente pendant Early Access 

# Construire le corpus

## Combiner tous les commentaires
Les commentaires extraits sont répartis dans plusieurs fichiers CSV. Pour construire un corpus complet, nous procédons à la fusion de tous les fichiers dans un seul fichier maître. La procédure est automatisée à l’aide du module `pandas`.

## Filtrer les commentaires
caractère non-sense
heure de jouer

## Preprocessing
stopword, lower, lemmatisation, ponctuation...

# Entrainement

## annotation -> 1 doc = 1 classe (pas besion)

## decoupage (dev / train / test)

## vectorisation (countvect / tf-idf)

## entrainement (train)

## evaluation (test)