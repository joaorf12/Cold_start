import pandas as pd
import ast

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
        # if True:
        #     df_generos_unificado.to_csv('df_generos_unificado.csv', index=False)
        #     print("DataFrame de gêneros unificado salvo em 'df_generos_unificado.csv'.")

        return df_generos_unificado

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique se os arquivos estão no caminho correto: {e}")
        return None
