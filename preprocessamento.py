import pandas as pd
import ast
import json
import os
import time

def criar_df_generos_unificado():
    """
    Cria um DataFrame unificado de gêneros a partir de data_w_genres.csv
    e dataset_spotify_track.csv.
    """
    try:
        # Carregar e processar o primeiro dataset de gêneros
        df_generos_1 = pd.read_csv('./datasets/data_w_genres.csv')
        df_generos_1 = df_generos_1[['artists', 'genres']].copy()

        # Limpar a coluna 'genres' (de string para lista)
        df_generos_1['genres'] = df_generos_1['genres'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

        # Agrupar gêneros por artista
        df_generos_1_grouped = df_generos_1.groupby('artists')['genres'].apply(
            lambda x: list(set([item for sublist in x for item in sublist]))
        ).reset_index()

        # Carregar e processar o segundo dataset de gêneros
        df_generos_2 = pd.read_csv('./datasets/dataset_spotify_track.csv')
        df_generos_2 = df_generos_2[['artists', 'track_genre']].copy()

        # Remover a vírgula para tratar múltiplos artistas
        df_generos_2['artists'] = df_generos_2['artists'].str.replace(r"\[|\]|'", '', regex=True)
        df_generos_2['artists'] = df_generos_2['artists'].str.split(';')

        # Expandir as linhas para que cada artista fique em uma linha separada
        df_generos_2 = df_generos_2.explode('artists')
        df_generos_2['artists'] = df_generos_2['artists'].str.strip()

        # Agrupar gêneros por artista
        df_generos_2_grouped = df_generos_2.groupby('artists')['track_genre'].apply(
            lambda x: list(set(x.dropna()))
        ).reset_index().rename(columns={'track_genre': 'genres'})

        # Mesclar os dois DataFrames de gêneros
        df_generos_unificado = pd.merge(
            df_generos_1_grouped, df_generos_2_grouped, on='artists', how='outer'
        )

        # Unificar as listas de gêneros
        def unify_genres(row):
            genres_list = []
            if isinstance(row['genres_x'], list):
                genres_list.extend(row['genres_x'])
            if isinstance(row['genres_y'], list):
                genres_list.extend(row['genres_y'])
            return list(set(genres_list))

        df_generos_unificado['genres'] = df_generos_unificado.apply(unify_genres, axis=1)
        df_generos_unificado = df_generos_unificado.drop(columns=['genres_x', 'genres_y'])

        # Padronizar nomes de artistas
        df_generos_unificado['artists_clean'] = df_generos_unificado['artists'].str.lower().str.replace(r'[^\w\s]', '',
                                                                                                        regex=True).str.strip()
        if True:
            df_generos_unificado.to_csv('df_generos_unificado.csv', index=False)
            print("DataFrame de gêneros unificado salvo em 'df_generos_unificado.csv'.")

        return df_generos_unificado

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique se os arquivos estão no caminho correto: {e}")
        return None


def criar_e_salvar_dicionario_generos(df_musicas_com_generos):
    """
    Cria um dicionário mapeando o ID de cada música para a lista de gêneros do artista
    e o salva em um arquivo JSON.
    """
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando a criação do dicionário de gêneros...")

    # Garante que a coluna 'artist_genres' existe e é uma lista válida
    if 'artist_genres' not in df_musicas_com_generos.columns:
        print("Erro: Coluna 'artist_genres' não encontrada no DataFrame de entrada.")
        return

    # Limpeza e preparação da coluna de gêneros:
    # 1. Preenche NaNs com uma lista vazia
    df_musicas_com_generos['artist_genres'] = df_musicas_com_generos['artist_genres'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    generos_dict = {}
    total_musicas = len(df_musicas_com_generos)

    print(f"[{time.strftime('%H:%M:%S')}] Processando {total_musicas} músicas...")

    # Cria o dicionário: mapeia 'id' da música para a lista de 'artist_genres'
    for _, row in df_musicas_com_generos.iterrows():
        track_id = row['id']
        generos_dict[track_id] = row['artist_genres']

    # --- Salvando o arquivo JSON ---
    nome_arquivo = "./generos_por_musica_dict.json"
    print(f"[{time.strftime('%H:%M:%S')}] Salvando o dicionário em {nome_arquivo}...")

    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(generos_dict, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Dicionário criado e salvo com sucesso em {nome_arquivo}.")
    if os.path.exists(nome_arquivo):
        print(f"Tamanho do arquivo gerado: {os.path.getsize(nome_arquivo) / (1024 * 1024):.2f} MB.")
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos.")


def unificar_e_salvar_generos(
    caminho_musicas='./datasets/data.csv',
    caminho_generos_w='./datasets/data_w_genres.csv',
    caminho_generos_track='./datasets/dataset_spotify_track.csv',
    caminho_artists='./datasets/artists.csv',
    caminho_data_by_artist='./datasets/data_by_artist.csv'
):
    """
    Função principal de pré-processamento que unifica dados de diferentes fontes,
    limpa e salva um dicionário mapeando o ID da música para os gêneros do artista.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando o pré-processamento dos dados de gênero...")

    try:
        df_musicas = pd.read_csv(caminho_musicas)
        df_generos_1 = pd.read_csv(caminho_generos_w)
        df_generos_2 = pd.read_csv(caminho_generos_track)
        df_artists = pd.read_csv(caminho_artists, sep='\t')
        df_data_by_artist = pd.read_csv(caminho_data_by_artist)
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique os caminhos: {e}")
        return None

    def clean_artist_name(name):
        return name.astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"[{time.strftime('%H:%M:%S')}] Unificando as bases de dados de gênero...")

    df_generos_1['genres'] = df_generos_1['genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df_generos_1['artists_clean'] = clean_artist_name(df_generos_1['artists'])
    df_generos_1_grouped = df_generos_1.groupby('artists_clean')['genres'].apply(
        lambda x: list(set([item for sublist in x for item in sublist]))
    ).reset_index()

    df_generos_2['artists'] = df_generos_2['artists'].str.replace(r"\[|\]|'", '', regex=True)
    df_generos_2['artists'] = df_generos_2['artists'].str.split(';')
    df_generos_2 = df_generos_2.explode('artists')
    df_generos_2['artists_clean'] = clean_artist_name(df_generos_2['artists'])
    df_generos_2_grouped = df_generos_2.groupby('artists_clean')['track_genre'].apply(
        lambda x: list(set(x.dropna()))
    ).reset_index().rename(columns={'track_genre': 'genres'})

    df_generos_unificado = pd.merge(
        df_generos_1_grouped, df_generos_2_grouped, on='artists_clean', how='outer', suffixes=('_x', '_y')
    )
    def unify_genres(row):
        genres_x = row['genres_x'] if isinstance(row['genres_x'], list) else []
        genres_y = row['genres_y'] if isinstance(row['genres_y'], list) else []
        return list(set(genres_x + genres_y))

    df_generos_unificado['genres'] = df_generos_unificado.apply(unify_genres, axis=1)
    df_generos_unificado.drop(columns=['genres_x', 'genres_y'], inplace=True)

    print(f"[{time.strftime('%H:%M:%S')}] Mesclando gêneros com as bases de artistas e músicas...")
    df_artists['name_clean'] = clean_artist_name(df_artists['name'])
    df_data_by_artist['artists_clean'] = clean_artist_name(df_data_by_artist['artists'])
    df_musicas['artists_clean'] = clean_artist_name(df_musicas['artists'])

    df_artists_com_generos = pd.merge(df_artists, df_generos_unificado, left_on='name_clean', right_on='artists_clean', how='left')
    df_artists_com_generos.rename(columns={'genres': 'artist_genres'}, inplace=True)

    df_musicas_com_generos = pd.merge(df_musicas, df_artists_com_generos[['name_clean', 'artist_genres']], left_on='artists_clean', right_on='name_clean', how='left')
    df_musicas_com_generos['artist_genres'] = df_musicas_com_generos['artist_genres'].fillna('').apply(lambda x: x if isinstance(x, list) else [])

    print(f"[{time.strftime('%H:%M:%S')}] Removendo músicas com gêneros vazios...")
    num_antes = len(df_musicas_com_generos)
    df_musicas_com_generos = df_musicas_com_generos[df_musicas_com_generos['artist_genres'].apply(len) > 0].copy()
    num_depois = len(df_musicas_com_generos)
    print(f"[{time.strftime('%H:%M:%S')}] {num_antes - num_depois} músicas removidas. Restam {num_depois} músicas.")

    # Altera para salvar em CSV
    nome_arquivo_saida = "./datasets/musicas_com_generos.csv"
    df_musicas_com_generos.to_csv(nome_arquivo_saida, index=False)

    print(f"[{time.strftime('%H:%M:%S')}] Base de dados de músicas com gêneros salva em '{nome_arquivo_saida}'.")
    return df_musicas_com_generos


if __name__ == '__main__':
    unificar_e_salvar_generos()