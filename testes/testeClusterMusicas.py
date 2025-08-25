import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from openai import RateLimitError
from visualizacao import plotar_clusters, print_playlist_bonita
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from chamadasChatGPT import chamada_api_retry, gerar_novo_usuario_aleatorio, criar_persona_chatgpt, obter_reacao_persona

# ==================== Configurações ====================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==================== Funções auxiliares ====================
def plot_clusters_heatmap(data, labels, features, titulo="Média das Features por Cluster"):
    """
    Plota um heatmap das médias das features para cada cluster.
    """
    data_copy = data.copy()
    data_copy['cluster'] = labels
    cluster_means = data_copy.groupby('cluster')[features].mean()

    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Valor médio'})
    plt.title(titulo, fontsize=16)
    plt.ylabel("Cluster")
    plt.xlabel("Feature")
    plt.show()

def gerar_recomendacao_completa_musicas(musicas_base, musicas_com_generos, musical_features,
                                        preference_map, num_recommendations_needed=10,
                                        teste_rapido=False, amostra_teste=2000):
    """
    Clusteriza músicas diretamente e gera recomendações para um novo usuário aleatório.
    - teste_rapido: se True, usa apenas uma amostra para escolher n_clusters.
    - amostra_teste: tamanho da amostra usada em teste_rapido.
    """

    # --- Limpeza e padronização ---
    musicas_base['artists_clean'] = musicas_base['artists'].str.lower().str.replace(r'[^\w\s]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
    musicas_com_generos['artists_clean'] = musicas_com_generos['artists'].str.lower().str.replace(r'[^\w\s]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
    musicas_com_generos['genres'] = musicas_com_generos['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x,str) and x.startswith('[') else ([] if pd.isna(x) else [str(x)])
    )

    aggregated_genres = musicas_com_generos.groupby('artists_clean')['genres'].apply(
        lambda x: list(set(g for sublist in x for g in sublist))
    ).reset_index().rename(columns={'genres':'artist_genres'})

    data_with_genres = pd.merge(musicas_base, aggregated_genres, on='artists_clean', how='left')
    data_with_genres['artist_genres'] = data_with_genres['artist_genres'].fillna('').apply(lambda x: x if isinstance(x,list) else [])

    # --- Preparação das features ---
    X = data_with_genres[musical_features].fillna(data_with_genres[musical_features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Amostra para teste rápido ---
    if teste_rapido:
        np.random.seed(42)
        idx_amostra = np.random.choice(len(X_scaled), min(amostra_teste, len(X_scaled)), replace=False)
        X_scaled_teste = X_scaled[idx_amostra]
    else:
        X_scaled_teste = X_scaled

    # --- Escolha do número de clusters ---
    cluster_range = range(2, 50)
    diversity_scores = []

    print("🔹 Clusterizando músicas para escolher o melhor número de clusters...")
    for n_clusters in tqdm(cluster_range):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        labels = km.fit_predict(X_scaled_teste)
        sil_score = silhouette_score(X_scaled_teste, labels)
        centroids = km.cluster_centers_
        pairwise_distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
        mean_distance = np.mean(pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)])
        diversity_scores.append(0.5 * sil_score + 0.5 * mean_distance)

    best_idx = np.argmax(diversity_scores)
    n_clusters_final = cluster_range[best_idx]
    print(f"✅ Número de clusters escolhido: {n_clusters_final}")

    # --- Clusterização final no dataset completo ---
    kmeans = KMeans(n_clusters=n_clusters_final, random_state=42, n_init='auto')
    data_with_genres['cluster'] = kmeans.fit_predict(X_scaled)

    # --- Plot dos clusters usando PCA ---
    plot_clusters_heatmap(data_with_genres, data_with_genres['cluster'].values, musical_features)

    # --- Novo usuário e persona ---
    novo_usuario = chamada_api_retry(gerar_novo_usuario_aleatorio)
    persona_gerada = chamada_api_retry(criar_persona_chatgpt, novo_usuario, preference_map)

    new_user_data = {}
    for feat in musical_features:
        if feat in novo_usuario:
            val = novo_usuario[feat]
            new_user_data[feat] = preference_map[feat][val.lower()] if isinstance(val, str) else val
        else:
            new_user_data[feat] = data_with_genres[feat].mean()
    new_user_vector = np.array([new_user_data[feat] for feat in musical_features]).reshape(1, -1)

    # --- Seleciona clusters mais próximos ---
    distances = kmeans.transform(new_user_vector)[0]
    nearest_cluster_indices = np.argsort(distances)[:3]

    # --- Filtragem de músicas ---
    musicas_filtradas = data_with_genres[data_with_genres['cluster'].isin(nearest_cluster_indices)].copy()
    pop_threshold = musicas_filtradas['popularity'].quantile(0.5)
    musicas_filtradas = musicas_filtradas[musicas_filtradas['popularity'] >= pop_threshold]

    persona_generos = list(set(re.findall(
        r'\b(rock|pop|hip hop|rap|r&b|country|soul|indie|electropop|synthpop)\b', persona_gerada.lower()
    )))
    if persona_generos:
        musicas_filtradas_por_genero = musicas_filtradas[musicas_filtradas['artist_genres'].apply(
            lambda gs: any(g in gs for g in persona_generos)
        )]
        if len(musicas_filtradas_por_genero) >= num_recommendations_needed:
            musicas_filtradas = musicas_filtradas_por_genero

    musicas_filtradas['distancia_perfil'] = np.linalg.norm(musicas_filtradas[musical_features].values - new_user_vector, axis=1)
    musicas_filtradas = musicas_filtradas.sort_values(by=['distancia_perfil', 'popularity'], ascending=[True, False])

    # --- Limitar 3 músicas por artista ---
    playlist_final = []
    contador_artistas = {}
    for _, row in musicas_filtradas.iterrows():
        artista = row['artists']
        if contador_artistas.get(artista, 0) < 3:
            playlist_final.append(row)
            contador_artistas[artista] = contador_artistas.get(artista, 0) + 1
        if len(playlist_final) >= num_recommendations_needed:
            break

    musicas_recomendadas_final = pd.DataFrame(playlist_final)[['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features]

    # --- Reação da persona ---
    if not musicas_recomendadas_final.empty:
        nomes_musicas = [f"{row['name']} - {row['artists']}" for _, row in musicas_recomendadas_final.iterrows()]
        reacao = chamada_api_retry(obter_reacao_persona, nomes_musicas, persona_gerada)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    return novo_usuario, persona_gerada, musicas_recomendadas_final, reacao

# ==================== Uso da função clusterizando músicas ====================
musical_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

preference_map = {
    'danceability': {'alto': 0.95, 'medio': 0.5, 'baixo': 0.05},
    'energy': {'alto': 0.95, 'medio': 0.5, 'baixo': 0.05},
    'loudness': {'alto': -2.0, 'medio': -15.0, 'baixo': -45.0},
    'valence': {'alto': 0.95, 'medio': 0.5, 'baixo': 0.05},
    'acousticness': {'alto': 0.05, 'medio': 0.4, 'baixo': 0.95},
    'instrumentalness': {'alto': 0.95, 'medio': 0.4, 'baixo': 0.0},
    'liveness': {'alto': 0.95, 'medio': 0.4, 'baixo': 0.01},
    'speechiness': {'alto': 0.9, 'medio': 0.25, 'baixo': 0.01},
    'tempo': {'alto': 190.0, 'medio': 110.0, 'baixo': 50.0}
}

# Carregando datasets
musicas_base = pd.read_csv('../datasets/data.csv')
musicas_com_generos = pd.read_csv('../datasets/data_w_genres.csv')

novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa_musicas(
    musicas_base,
    musicas_com_generos,
    musical_features,
    preference_map,
    num_recommendations_needed=10,
    teste_rapido=True,
    amostra_teste=2000
)

# Exibindo resultados
print("Novo usuário:", novo_usuario)
print("Persona gerada:", persona)
print_playlist_bonita(playlist)
if reacao:
    print("\n📝 Reação da persona:")
    print(reacao)
