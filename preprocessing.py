import pandas as pd

# Conbination de tous les commentaires

def combine(folder_path:str) -> pd.DataFrame:
    import os
    from glob import glob
    
    # récupérer tous les fichiers csv
    csv_files = glob(os.path.join(folder_path, '*.csv'))
    # combiner tous les fichiers
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    # sauvgarder 
    combined_df.to_csv('data/Steam_Reviews.csv', index=False)
    print(f'Conbiner {len(csv_files)} fichiers, commentaires totales : {len(combined_df)}')
    
    return combined_df

# Filtrage (garder seulment les commentaires françaises)

def detect_language(text:str) -> str:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

def filter(df:pd.DataFrame) -> pd.DataFrame:
    df['lang'] = df['comment'].apply(detect_language)
    df_fr = df[df['lang'] == 'fr']
    
    df_fr.to_csv('data/Steam_Reviews_fr.csv', index=False)
    print("fichier sauvgardé : Steam_Reviews_fr.csv")

    return df_fr

# Preprocessing (Optinal)

def lowercase(text:str) -> str:
    text = text.lower()
    return text

def url(text:str) -> str:
    import re
    
    # enlever la balise html
    text = re.sub(r'<.*?>', '', text)
    #enlever l'url
    text = re.sub(r'http\S+|www\S+', '', text)
    
    return text

def ponct(text:str) -> str:
    import string
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Organiser le datdaset (on focaliser sur le tag et comment)

def map_to_categorical(df:pd.DataFrame) -> pd.DataFrame:
    import json
    
    df['label'] = pd.Categorical(df.tag, ordered=True).codes
    
    label2Index = {row['tag']: row['label'] for idx, row in df.iterrows()}
    index2label = {row['label']: row['tag'] for idx, row in df.iterrows()}
    with open("label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({
            "label2Index": label2Index,
            "index2label": index2label
        }, f, ensure_ascii=False, indent=2)
    
    df.rename(columns={'label': 'labels', 'comment': 'text'}, inplace=True)
    return df[['text', 'labels']]

# Diviser le dataset en test, val et test

def split(df:pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    # première division ：test (80%) et temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])

    # dexième division：parmis temp, val (10%) et test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["labels"])

    # sauvgarder
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("Succès : train.csv / val.csv / test.csv")

    return train_df, val_df, test_df

if __name__ == '__main__':
    path = 'data'
    df_all = combine(path)
    df_fr = filter(df_all)
    df = map_to_categorical(df_fr)
    split(df)