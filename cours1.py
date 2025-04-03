import re
import spacy
import nltk
from nltk import word_tokenize, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from glob import glob
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
stop_words_fr = stopwords.words('french')

def tokenize(text):
    words = word_tokenize(text)
    tags = ne_chunk(nltk.pos_tag(words))
    return words, tags


def extract_named_entities(text):
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text)
    entities = []
    for ent in doc.ents :
        entities.append((ent.text, ent.label_))
    return entities


def lowercase(text) :
    text =text.lower()
    print(text)

def remove_stopword(text) :
    with open("Stopwords.txt") as lexique :
        lexique = lexique.read()
        text = text.split(" ")
        for elem in text:
            if elem in lexique :
                pass
            else :
                print(elem)

def remove_punc(text):
    punctuations = ['.', ',', ';', ':', '!', '?', '«', '»', '(', ')','"',' – ']
    for p in punctuations:
        text = text.replace(p, '')
    return text


def remove_url(text):
    url_patterns = [r'https?://\S+', r'www\.\S+']
    for pattern in url_patterns:
        text = re.sub(pattern, '', text)
    return text

def n_gram(text, n):
    words = text.split()
    for i in range(len(words) - n + 1):
        for x in range(n):
            print(words[i + x], end=' ')

        print("\n \n")

    return

with open("cours1.txt") as text :
    text_list = []
    text = text.read()
    text_list.append(text)
    #lowercase(text)
    #remove_stopword(text)
    #print(remove_punc(text))
    #print(remove_url(text))
    #print(n_gram(text, int(input("Taille n_gram?"))))
    #print(extract_named_entities(text))
    #print(tokenize(text))
    vectorizer = CountVectorizer() #strip_accents='ascii'/ preprocessor (Callable)
    X = vectorizer.fit_transform(text_list)
    print(vectorizer.get_feature_names_out())



    """"If a string, it is passed to _check_stop_list and the appropriate stop list is returned. ‘english’ is currently the only supported string value. > NLTK"""
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_fr, ngram_range=(1, 2), max_features=50) #N features//token_patternstr, default=r”(?u)\b\w\w+\b”
    X_tfidf = tfidf_vectorizer.fit_transform(text_list)

    print("TfidfVectorizer voc :\n")
    print(tfidf_vectorizer.get_feature_names_out())
    print(X_tfidf.toarray())
#Faire deux fichier séparés
