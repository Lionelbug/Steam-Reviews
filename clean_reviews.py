import pandas as pd
import os
from glob import glob

# Conbination de tous les commentaires
folder_path = 'data/raw/'
csv_files = glob(os.path.join(folder_path, '*.csv'))

df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df.to_csv('data/clean/Steam_Reviews.csv', index=False)

print(f'Conbiner {len(csv_files)} fichiers, commentaires totales : {len(combined_df)}')

# Filtrage

# Preprocessing

