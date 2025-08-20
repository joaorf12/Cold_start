import pandas as pd
import numpy as np
import re
import ast

# Função de limpeza de nomes
def limpar_nome(nome):
    nome = str(nome).lower()
    # Remoção manual de acentos para caracteres comuns em português
    nome = nome.replace('á', 'a').replace('a', 'a').replace('ã', 'a').replace('â', 'a')
    nome = nome.replace('é', 'e').replace('ê', 'e')
    nome = nome.replace('í', 'i')
    nome = nome.replace('ó', 'o').replace('õ', 'o').replace('ô', 'o')
    nome = nome.replace('ú', 'u')
    nome = nome.replace('ç', 'c')
    nome = re.sub(r'[^\w\s]', '', nome)  # remove pontuação
    nome = re.sub(r'\s+', ' ', nome).strip()  # remove espaços extras
    return nome

# ==============================================================================
# BLOCO 1: CARREGAR E PREPARAR OS DATASETS
# ==============================================================================

# Carregar data.csv e data_by_artist.csv
dataset_musicas_1 = pd.read_csv("./datasets/data.csv")
dataset_musicas_2 = pd.read_csv("./datasets/dataset_spotify_track.csv")
data_by_artist = pd.read_csv("./datasets/data_by_artist.csv")

# Renomear colunas do dataset 2 para que correspondam ao dataset 1
dataset_musicas_2.rename(columns={
    'track_id': 'id',
    'track_name': 'name'
}, inplace=True)

# Remover a coluna `genres` do dataset 2 para evitar conflitos na concatenação inicial
# O gênero será tratado posteriormente
dataset_musicas_2 = dataset_musicas_2.drop(columns=['track_genre'], errors='ignore')

# Gerar a nova coluna 'artists_clean' em ambos os DataFrames
dataset_musicas_1['artists_clean'] = dataset_musicas_1['artists'].apply(limpar_nome)
dataset_musicas_2['artists_clean'] = dataset_musicas_2['artists'].apply(limpar_nome)
data_by_artist['artists_clean'] = data_by_artist['artists'].apply(limpar_nome)

# ==============================================================================
# BLOCO 2: MESCLAR OS DOIS DATASETS DE MÚSICAS
# ==============================================================================

# Garantir que os dois DataFrames de música tenham as mesmas colunas antes de mesclar
# Isso previne erros de alinhamento
colunas_comuns = list(set(dataset_musicas_1.columns) & set(dataset_musicas_2.columns))
dataset_musicas_1_uniformizado = dataset_musicas_1[colunas_comuns]
dataset_musicas_2_uniformizado = dataset_musicas_2[colunas_comuns]

# Concatenar os dois DataFrames. 'ignore_index=True' garante um novo índice contínuo.
dataset_final = pd.concat([dataset_musicas_1_uniformizado, dataset_musicas_2_uniformizado], ignore_index=True)

# Remover duplicatas com base no ID da música
# O keep='first' mantém a primeira ocorrência da música encontrada
dataset_final.drop_duplicates(subset=['id'], keep='first', inplace=True)

print(f"Dataset 1 tinha {len(dataset_musicas_1)} músicas.")
print(f"Dataset 2 tinha {len(dataset_musicas_2)} músicas.")
print(f"Dataset final (mesclado e sem duplicatas) tem {len(dataset_final)} músicas.")

# Salvar o novo DataFrame final no arquivo `data.csv`
dataset_final.to_csv("./datasets/data.csv", index=False)
data_by_artist.to_csv("./datasets/data_by_artist.csv", index=False)


# ==============================================================================
# BLOCO 3: RECRIAÇÃO DOS OUTROS ARQUIVOS COM O NOVO DATASET
# ==============================================================================

# Criar tabela de artistas únicos a partir dos dados limpos do novo dataset final
# Usar a coluna `artists_clean` do `dataset_final`
unique_artists = dataset_final['artists_clean'].drop_duplicates().reset_index(drop=True)

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