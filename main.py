import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessamento import criar_df_generos_unificado
from visualizacao import plotar_clusters, print_playlist_bonita
from chamadasGemini import chamada_api_retry, gerar_novo_usuario_aleatorio, criar_persona_gemini, obter_reacao_persona


# ==================== Função principal ====================
def gerar_recomendacao_completa(interacoes_usuarios_artistas, dados_artistas,
                                caracteristicas_por_artistadata_by_artist, musicas_base,
                                musicas_com_generos, musical_features, label_cols,
                                preference_map, num_recommendations_needed=10):
    # Dicionário para mapear gêneros descritivos para gêneros do dataset
    generos_mapeamento = {
        'rock': ['rock', 'classic rock', 'hard rock', 'rock and roll', 'alternative rock'],
        'pop': ['pop', 'indie pop', 'pop rock', 'synth-pop', 'electropop'],
        'eletrônica': ['edm', 'dance pop', 'electronic', 'house', 'dubstep', 'techno', 'trance'],
        'hip hop': ['hip hop', 'rap', 'trap', 'drill'],
        'r&b': ['r&b', 'soul'],
        'jazz': ['jazz', 'smooth jazz', 'swing'],
        'folk': ['folk', 'singer-songwriter'],
        'metal': ['metal', 'heavy metal', 'death metal', 'black metal', 'thrash metal'],
        'clássica': ['classical', 'symphony', 'orchestral'],
        'blues': ['blues', 'delta blues', 'electric blues'],
        'indie': ['indie pop', 'indie rock', 'indie folk'],
        'country': ['country', 'alt-country'],
        'dance': ['dance pop', 'edm', 'house'],
        'disco': ['disco', 'funk'],
        'reggae': ['reggae', 'dub'],
        'latina': ['latin', 'reggaeton', 'salsa', 'bachata'],
        'k-pop': ['k-pop'],
        'j-pop': ['j-pop'],
    }

    def obter_generos_mapeados(generos_persona):
        generos_traduzidos = set()
        for genero_persona in generos_persona:
            genero_lower = genero_persona.lower()
            for chave, valores in generos_mapeamento.items():
                if chave in genero_lower:
                    generos_traduzidos.update(valores)
        return list(generos_traduzidos)

    # --- Limpeza e padronização ---
    dfs_cols = [
        (dados_artistas, 'name', 'name_clean'),
        (caracteristicas_por_artistadata_by_artist, 'artists', 'artists_clean'),
        (musicas_base, 'artists', 'artists_clean'),
    ]
    for df, col_in, col_out in dfs_cols:
        df[col_out] = df[col_in].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ',
                                                                                                 regex=True).str.strip()

    # Usar o DataFrame de gêneros unificado passado como argumento e corrigir o merge
    data_with_genres = pd.merge(musicas_base, musicas_com_generos, on='artists_clean', how='left')
    data_with_genres.rename(columns={'artists_x': 'artists', 'genres': 'artist_genres'}, inplace=True)
    data_with_genres.drop(columns=['artists_y'], inplace=True, errors='ignore')
    data_with_genres['artist_genres'] = data_with_genres['artist_genres'].fillna('').apply(
        lambda x: x if isinstance(x, list) else [])

    # --- Simulação de perfil ---
    np.random.seed(42)
    unique_users = interacoes_usuarios_artistas['userID'].unique()
    fake_users = pd.DataFrame({
        'userID': unique_users,
        'age': np.random.randint(15, 60, len(unique_users)),
        'gender': np.random.choice(['m', 'f'], len(unique_users)),
        'country': np.random.choice(['Brazil', 'USA', 'Germany', 'UK', 'Japan'], len(unique_users))
    })
    merged = pd.merge(interacoes_usuarios_artistas, fake_users, on='userID')
    top_artists = merged.loc[merged.groupby('userID')['weight'].idxmax()]
    top_artists = pd.merge(top_artists, dados_artistas[['id', 'name', 'name_clean']], left_on='artistID', right_on='id',
                           how='left')
    top_artists = pd.merge(top_artists, caracteristicas_por_artistadata_by_artist, left_on='name_clean',
                           right_on='artists_clean', how='left')

    # =========================
    # Escolha do número de clusters visando DIVERSIDADE
    # =========================
    X_features = ['age', 'gender', 'country'] + [f for f in musical_features if f in top_artists.columns]

    for feature in X_features:
        if feature not in ['age', 'gender', 'country']:
            top_artists[feature] = top_artists[feature].fillna(top_artists[feature].mean())

    encoders = {col: LabelEncoder().fit(top_artists[col]) for col in label_cols}
    for col in label_cols:
        top_artists[col] = encoders[col].transform(top_artists[col])

    X = top_artists[X_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_range = range(2, 50)
    diversity_scores = []

    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)
        centroids = km.cluster_centers_
        pairwise_distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
        mean_distance = np.mean(pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)])
        diversity_score = 0.5 * sil_score + 0.5 * mean_distance
        diversity_scores.append(diversity_score)

    best_idx = np.argmax(diversity_scores)
    n_clusters_final = cluster_range[best_idx]

    kmeans = KMeans(n_clusters=n_clusters_final, random_state=42, n_init='auto')
    top_artists['cluster'] = kmeans.fit_predict(X_scaled)

    print(f"Número de clusters escolhido visando diversidade: {n_clusters_final}")

    # --- Usuário e persona ---
    novo_usuario = chamada_api_retry(gerar_novo_usuario_aleatorio)
    persona_info = chamada_api_retry(criar_persona_gemini, novo_usuario, preference_map)
    persona_gerada = persona_info['persona_text']
    persona_generos = persona_info['persona_genres']

    # Adicionar a nova lógica de mapeamento de gêneros
    generos_para_filtro = obter_generos_mapeados(persona_generos)

    new_user_data = {}
    for key in ['age', 'gender', 'country']:
        if key in label_cols:
            new_user_data[key] = encoders[key].transform([novo_usuario[key]])[0]
        else:
            new_user_data[key] = novo_usuario[key]
    for feat in musical_features:
        if feat in novo_usuario:
            val = novo_usuario[feat]
            if isinstance(val, str):
                new_user_data[feat] = preference_map[feat][val.lower()]
            else:
                new_user_data[feat] = val
        else:
            new_user_data[feat] = top_artists[feat].mean()
    new_user_df = pd.DataFrame([new_user_data])[X_features]
    new_user_scaled = scaler.transform(new_user_df)

    # --- Clusters mais próximos ---
    distances = kmeans.transform(new_user_scaled)[0]
    min_distance = distances.min()
    threshold = min_distance * 1.2
    nearest_cluster_indices = np.where(distances <= threshold)[0]

    print(f"\nO novo usuário está relacionado aos clusters: {nearest_cluster_indices}")
    print(f"O usuário foi aderido a {len(nearest_cluster_indices)} clusters.")

    # --- Filtragem de músicas pelos clusters ---
    artistas_clusters = top_artists[top_artists['cluster'].isin(nearest_cluster_indices)]['name_clean'].unique()
    musicas_filtradas = data_with_genres[data_with_genres['artists_clean'].isin(artistas_clusters)].copy()
    if musicas_filtradas.empty:
        musicas_filtradas = data_with_genres.copy()

    # --- Filtro de popularidade ---
    pop_threshold = musicas_filtradas['popularity'].quantile(0.5)
    musicas_filtradas = musicas_filtradas[musicas_filtradas['popularity'] >= pop_threshold]
    if musicas_filtradas.empty:
        musicas_filtradas = data_with_genres.copy()

    # --- Adicionar estratégia de fallback para gêneros ---
    if generos_para_filtro:
        # Tenta filtrar músicas dos clusters pelos gêneros da persona (agora mapeados!)
        musicas_por_genero = musicas_filtradas[musicas_filtradas['artist_genres'].apply(
            lambda gs: any(g in gs for g in generos_para_filtro)
        )]

        if musicas_por_genero.empty or len(musicas_por_genero) < num_recommendations_needed / 2:
            print("INFO: O filtro de gênero falhou. Buscando em toda a base de dados por gêneros da persona...")

            # Novo dataframe, agora buscando em todo o conjunto de dados
            fallback_musicas = data_with_genres[data_with_genres['artist_genres'].apply(
                lambda gs: any(g in gs for g in generos_para_filtro)
            )].copy()

            # Filtro de popularidade na nova base
            if not fallback_musicas.empty:
                pop_threshold_fallback = fallback_musicas['popularity'].quantile(0.5)
                fallback_musicas = fallback_musicas[fallback_musicas['popularity'] >= pop_threshold_fallback]

            if not fallback_musicas.empty:
                musicas_filtradas = fallback_musicas
            else:
                print("INFO: Busca de fallback falhou. Usando músicas originais sem filtro de gênero.")

    # --- Ordenar por proximidade e limitar repetição de artista ---
    user_vector = np.array([new_user_data[feat] for feat in musical_features])
    musicas_filtradas['distancia_perfil'] = np.linalg.norm(
        musicas_filtradas[musical_features].values - user_vector, axis=1
    )
    musicas_filtradas['pontuacao_genero'] = 0.0
    if generos_para_filtro:
        musicas_filtradas['pontuacao_genero'] = musicas_filtradas['artist_genres'].apply(
            lambda gs: 1.0 if any(g in gs for g in generos_para_filtro) else -0.5
        )

    musicas_filtradas['distancia_ponderada'] = musicas_filtradas['distancia_perfil'] - musicas_filtradas[
        'pontuacao_genero']

    cols_necessarias = ['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features
    musicas_filtradas = musicas_filtradas.sort_values(by=['distancia_ponderada', 'popularity'], ascending=[True, False])

    # ADICIONE ESTA LINHA: Remove músicas duplicadas baseadas no 'id'
    musicas_filtradas.drop_duplicates(subset=['name'], keep='first', inplace=True)

    playlist_final = []
    contador_artistas = {}
    for _, row in musicas_filtradas.iterrows():
        artista = row['artists']
        if contador_artistas.get(artista, 0) < 3:
            playlist_final.append(row.to_dict())
            contador_artistas[artista] = contador_artistas.get(artista, 0) + 1
        if len(playlist_final) >= num_recommendations_needed:
            break

    if len(playlist_final) < num_recommendations_needed:
        faltantes = num_recommendations_needed - len(playlist_final)
        amostras_extra = data_with_genres.sample(faltantes, random_state=42)
        playlist_final.extend(amostras_extra.to_dict('records'))

    musicas_recomendadas_final = pd.DataFrame(playlist_final)[cols_necessarias]

    if not musicas_recomendadas_final.empty:
        nomes_musicas = [
            f"{row['name']} - {row['artists']}"
            for _, row in musicas_recomendadas_final.iterrows()
        ]
        reacao = chamada_api_retry(obter_reacao_persona, nomes_musicas, persona_gerada, novo_usuario)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    return novo_usuario, persona_gerada, musicas_recomendadas_final, reacao


# --- Uso da função ---
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

label_cols = ['gender', 'country']

# Passo 1: Crie o DataFrame unificado de gêneros
df_generos = criar_df_generos_unificado()

# Verifique se o DataFrame foi criado com sucesso antes de continuar
if df_generos is not None:
    # Passo 2: Chame a função principal, passando o novo DataFrame
    novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa(
        pd.read_csv('./datasets/user_artists.csv', sep='\t'),
        pd.read_csv('./datasets/artists.csv', sep='\t'),
        pd.read_csv('./datasets/data_by_artist.csv'),
        pd.read_csv('./datasets/data.csv'),
        df_generos,
        musical_features,
        label_cols,
        preference_map,
        num_recommendations_needed=8
    )

    print("Novo usuário:", novo_usuario)
    print("Persona gerada:", persona)
    print_playlist_bonita(playlist)
    if reacao:
        print("\n📝 Reação da persona:")
        print(reacao)