import pandas as pd
import numpy as np
import re

# Função de limpeza de nomes
def limpar_nome(nome):
    nome = str(nome).lower()
    # Remoção manual de acentos para caracteres comuns em português
    nome = nome.replace('á', 'a').replace('à', 'a').replace('ã', 'a').replace('â', 'a')
    nome = nome.replace('é', 'e').replace('ê', 'e')
    nome = nome.replace('í', 'i')
    nome = nome.replace('ó', 'o').replace('õ', 'o').replace('ô', 'o')
    nome = nome.replace('ú', 'u')
    nome = nome.replace('ç', 'c')
    nome = re.sub(r'[^\w\s]', '', nome)  # remove pontuação
    nome = re.sub(r'\s+', ' ', nome).strip()  # remove espaços extras
    return nome

# Carregar data.csv e data_by_artist.csv
data = pd.read_csv("./datasets/data.csv")
data_by_artist = pd.read_csv("./datasets/data_by_artist.csv")


# Gerar a nova coluna 'artists_clean' em ambos os DataFrames
# Esta coluna conterá os nomes dos artistas em formato limpo
data['artists_clean'] = data['artists'].apply(limpar_nome)
data_by_artist['artists_clean'] = data_by_artist['artists'].apply(limpar_nome)

# Salvar os DataFrames de volta com a nova coluna
# Isso garante que 'main.py' pode carregar os dados já limpos
data.to_csv("./datasets/data.csv", index=False)
data_by_artist.to_csv("./datasets/data_by_artist.csv", index=False)


# Criar tabela de artistas únicos a partir dos dados limpos
unique_artists = data['artists_clean'].drop_duplicates().reset_index(drop=True)

artists_df = pd.DataFrame({
    'id': unique_artists.index + 1,
    'name': unique_artists # 'name' aqui já são os nomes limpos
})

# Salvar como artists.csv
artists_df.to_csv("./datasets/artists.csv", sep='\t', index=False)


# Parâmetros de simulação para user_artists.csv
np.random.seed(42)
num_users = 100
num_interactions = 1000

user_ids = np.random.randint(1, num_users + 1, size=num_interactions)
artist_ids = np.random.choice(artists_df['id'], size=num_interactions)
weights = np.random.randint(1, 100, size=num_interactions)

user_artists_df = pd.DataFrame({
    'userID': user_ids,
    'artistID': artist_ids,
    'weight': weights
})

# Salvar como user_artists.csv
user_artists_df.to_csv("./datasets/user_artists.csv", sep='\t', index=False)