import pandas as pd
import numpy as np
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from visualizacao import print_playlist_bonita
from mapeamento_generos import mapear_generos_artista
from chamadasGemini import (
    chamada_api_retry,
    gerar_novo_usuario_aleatorio,
    criar_persona_gemini,
    obter_reacao_persona,
)
import warnings
import ast
from preprocessamento import unificar_e_salvar_generos

# Ignorar FutureWarning para manter o output limpo
warnings.filterwarnings('ignore', category=FutureWarning)

# === Função para avaliar qualidade do clustering ===
def silhouette_diversidade(X, labels, alpha=0.7):
    # ignora clusters inválidos (se houver só 1 cluster)
    if len(set(labels)) <= 1 or len(set(labels)) >= len(X):
        return -1

    # Ignora clusters de ruído
    valid_labels = labels[labels != -1]
    if len(valid_labels) < 2:
        return -1

    try:
        sil = silhouette_score(X[labels != -1], valid_labels)
    except:
        sil = 0

    # diversidade = proporção de clusters relevantes (>2% dos pontos)
    unique, counts = np.unique(valid_labels, return_counts=True)
    proporcoes = counts / counts.sum()
    diversidade = np.sum(proporcoes > 0.02) / len(unique)

    return alpha * sil + (1 - alpha) * diversidade

# === Ajusta clusters pequenos juntando com vizinho mais próximo ===
def ajustar_clusters(df, min_size=10):
    cluster_counts = df['cluster'].value_counts()
    clusters_pequenos = cluster_counts[cluster_counts < min_size].index

    if len(clusters_pequenos) == 0:
        return df

    df_pequenos = df[df['cluster'].isin(clusters_pequenos)].copy()

    # usa só colunas numéricas do próprio df
    X_pequenos = df_pequenos.select_dtypes(include=['float64', 'int64'])

    from sklearn.cluster import KMeans
    n_clusters = max(1, len(df_pequenos) // min_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    novas_labels = kmeans.fit_predict(X_pequenos)

    maior_cluster_id = df['cluster'].max() + 1
    df_pequenos['cluster'] = novas_labels + maior_cluster_id

    df_final = pd.concat([df[~df['cluster'].isin(clusters_pequenos)], df_pequenos], ignore_index=True)
    return df_final

def gerar_perfis_aleatorios(num_perfis, musical_features, preference_map):
    """
    Gera um DataFrame com perfis de usuários aleatórios, incluindo dados
    demográficos e preferências musicais simuladas.
    """
    data = {
        'userID': range(1, num_perfis + 1),
        'age': np.random.randint(15, 60, num_perfis),
        'gender': np.random.choice(['m', 'f'], num_perfis, p=[0.5, 0.5]),
        'country': np.random.choice(['Brazil', 'USA', 'Germany', 'UK', 'Japan'], num_perfis)
    }

    # Simula preferências musicais com base no preference_map
    for feat in musical_features:
        # Pondera a probabilidade de cada preferência
        pref_options = list(preference_map[feat].keys())
        # Cria uma lista de preferências para cada usuário
        preferences = np.random.choice(pref_options, num_perfis)
        # Converte as preferências em valores numéricos
        data[feat] = [preference_map[feat][p] for p in preferences]

    df = pd.DataFrame(data)
    return df

def filtrar_playlist(df_musicas, generos_mapeamento, persona, pesos, top_n=20, generos_indesejados=None):
    """
    Filtra a playlist de acordo com a persona:
    - Prioriza gêneros que batem com a persona
    - Ajusta por energy e tempo
    - Adiciona pontuação por popularidade
    - Mantém fallback caso não haja músicas suficientes nos clusters
    """
    if generos_indesejados is None:
        generos_indesejados = []

    df = df_musicas.copy()

    # === REMOÇÃO DE GÊNEROS INDESEJADOS ANTES DO FILTRO PRINCIPAL ===
    generos_indesejados_mapeados = [g.lower().strip() for g in generos_indesejados]

    df = df[
        ~df['genres_lower'].apply(lambda gs: any(g in generos_indesejados_mapeados for g in gs))
    ].copy()

    if len(df) < len(df_musicas):
        print(f"INFO: {len(df_musicas) - len(df)} músicas foram removidas devido a gêneros indesejados.")

    # ======================
    # 1. Score de popularidade
    # ======================
    df['popularity_score'] = df['popularity'] / 100

    # ======================
    # 2. Score de correspondência de gênero (mais flexível e agora mais preciso)
    # ======================
    generos_persona_mapeados = persona.get('generos', []) + persona.get('subgeneros', [])

    if not generos_persona_mapeados:
        df['genero_score_norm'] = 0
    else:
        def genero_score(musica_genres):
            score = 0
            generos_musica_limpos = [g.lower() for g in mapear_generos_artista(musica_genres)]

            # Conta quantas vezes um gênero da persona aparece nos gêneros da música
            # Pontuação por correspondência exata
            matches = [g for g in generos_musica_limpos if g in generos_persona_mapeados]
            score = len(matches)

            # Adiciona bônus por correspondência parcial (difflib)
            for g_persona in generos_persona_mapeados:
                if not any(g_persona in g for g in matches):  # se já não foi uma correspondência exata
                    close_match = difflib.get_close_matches(g_persona, generos_musica_limpos, n=1, cutoff=0.7)
                    if close_match:
                        score += 0.5  # Bônus para correspondência próxima

            return score

        df['genero_score'] = df['artist_genres'].apply(genero_score)

        max_score = len(generos_persona_mapeados) * 1.5  # Ponto extra para a correspondência próxima
        if max_score == 0:
            df['genero_score_norm'] = 0
        else:
            df['genero_score_norm'] = df['genero_score'] / max_score

    # ======================
    # 3. Score de energia/tempo e INSTRUMENTAL
    # ======================
    energia_map = {'baixo': 0.2, 'medio': 0.5, 'alto': 0.8}
    tempo_map = {'baixo': 60, 'medio': 100, 'alto': 140}
    instrumental_map = {'baixo': 0.05, 'medio': 0.4, 'alto': 0.95}
    speechiness_map = {'baixo': 0.05, 'medio': 0.25, 'alto': 0.9}
    liveness_map = {'baixo': 0.01, 'medio': 0.4, 'alto': 0.95}

    energy_persona = energia_map.get(persona.get('energy'), 0.5)
    tempo_persona = tempo_map.get(persona.get('tempo'), 110)
    instrumental_persona = instrumental_map.get(persona.get('instrumentalness'), 0.4)
    speechiness_persona = speechiness_map.get(persona.get('speechiness'), 0.25)
    liveness_persona = liveness_map.get(persona.get('liveness'), 0.5)

    df['energy_score'] = 1 - abs(
        df['energy'] - energy_persona) / 0.8
    df['tempo_score'] = 1 - abs(
        df['tempo'] - tempo_persona) / 140
    df['instrumentalness_score'] = 1 - abs(
        df['instrumentalness'] - instrumental_persona) / 0.95
    df['speechiness_score'] = 1 - abs(
        df['speechiness'] - speechiness_persona) / 0.9
    df['liveness_score'] = 1 - abs(
        df['liveness'] - liveness_persona) / 0.99

    # ======================
    # 4. Score final (Pesos ajustados dinamicamente)
    # ======================
    df['final_score'] = (
            df['genero_score_norm'] * pesos['genero'] +
            df['energy_score'] * pesos['energy'] +
            df['tempo_score'] * pesos['tempo'] +
            df['popularity_score'] * pesos['popularity'] +
            df['instrumentalness_score'] * pesos['instrumentalness'] +
            df['speechiness_score'] * pesos['speechiness'] +
            df['liveness_score'] * pesos['liveness']
    )

    # === LÓGICA DE PENALIDADE DE GÊNERO ===
    for _, row in df.iterrows():
        musica_generos = [g.lower() for g in mapear_generos_artista(row['artist_genres'])]

        if any(g in generos_indesejados_mapeados for g in musica_generos):
            df.loc[row.name, 'final_score'] = 0.01

        if not any(g in generos_persona_mapeados for g in musica_generos):
            df.loc[row.name, 'final_score'] *= 0.25 # Era 0.5. Agora, penaliza ainda mais.

    df_musicas_sorted = df.sort_values(by='final_score', ascending=False)
    df_musicas_sorted = df_musicas_sorted.drop_duplicates(subset=['artists'], keep='first')

    return df_musicas_sorted.head(top_n)

# ==================== Função para encontrar o K ideal para KMeans ====================
def encontrar_melhor_k(X, k_range=(32, 70)):
    melhor_k = 2
    melhor_score = -1

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)

        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > melhor_score:
                melhor_score = score
                melhor_k = k

    print(f"Melhor K encontrado (baseado no Silhouette Score): {melhor_k}")
    return melhor_k

# --- FUNÇÃO DE SELEÇÃO COM LÓGICA DE DIVERSIDADE ---
def selecionar_musicas_diversas(df_musicas, num_recommendations, artist_cap=1, genre_cap=2, min_genres=3):
    """
    Seleciona músicas com base no score final, mas com limite de artistas e gêneros.
    Também garante um mínimo de gêneros na playlist final.
    """
    final_playlist = pd.DataFrame()
    artists_count = {}
    genres_count = {}
    genres_in_playlist = set()
    musicas_disponiveis = df_musicas.copy()

    while len(final_playlist) < num_recommendations and not musicas_disponiveis.empty:
        # Pega a música com o maior score disponível
        musica = musicas_disponiveis.iloc[0]
        artist = musica['artists']
        musica_genres = [g.lower() for g in mapear_generos_artista(musica['artist_genres'])]

        # Verifica se o artista e os gêneros respeitam o limite
        artist_ok = artists_count.get(artist, 0) < artist_cap
        genres_ok = True
        for genre in musica_genres:
            if genres_count.get(genre, 0) >= genre_cap:
                genres_ok = False
                break

        if artist_ok and genres_ok:
            final_playlist = pd.concat([final_playlist, musica.to_frame().T], ignore_index=True)
            artists_count[artist] = artists_count.get(artist, 0) + 1
            for genre in musica_genres:
                genres_count[genre] = genres_count.get(genre, 0) + 1
                genres_in_playlist.add(genre)
            musicas_disponiveis = musicas_disponiveis.iloc[1:]
        else:
            musicas_disponiveis = musicas_disponiveis.iloc[1:]

    # Lógica de fallback para garantir min_genres
    if len(genres_in_playlist) < min_genres and len(final_playlist) < num_recommendations:
        print(f"AVISO: A playlist tem menos de {min_genres} gêneros. Buscando mais músicas para diversificar...")
        faltantes = num_recommendations - len(final_playlist)
        all_genres_ranked = df_musicas.copy()
        for _, row in all_genres_ranked.iterrows():
            musica_genres = [g.lower() for g in mapear_generos_artista(row['artist_genres'])]
            if not any(g in genres_in_playlist for g in musica_genres):
                final_playlist = pd.concat([final_playlist, row.to_frame().T], ignore_index=True)
                if len(final_playlist) >= num_recommendations:
                    break

    return final_playlist.head(num_recommendations)

# --- FUNÇÃO PARA BALANCEAR POPULARIDADE ---
def balancear_popularidade(df_musicas, num_recommendations, pesos_popularidade):
    df = df_musicas.copy().sort_values(by='final_score', ascending=False)
    total_musicas = len(df)

    # Classifica as músicas em tiers de popularidade
    df['popularity_tier'] = pd.qcut(df['popularity'], q=3, labels=['low', 'mid', 'top'])

    # Define a proporção de músicas a ser selecionada de cada tier
    # Ex: top 40%, mid 40%, low 20%
    num_top = int(num_recommendations * pesos_popularidade['top'])
    num_mid = int(num_recommendations * pesos_popularidade['mid'])
    num_low = num_recommendations - num_top - num_mid

    playlist_top = df[df['popularity_tier'] == 'top'].head(num_top)
    playlist_mid = df[df['popularity_tier'] == 'mid'].head(num_mid)
    playlist_low = df[df['popularity_tier'] == 'low'].head(num_low)

    final_playlist = pd.concat([playlist_top, playlist_mid, playlist_low]).sort_values(by='final_score',
                                                                                       ascending=False)

    return final_playlist.drop_duplicates(subset=['id']).head(num_recommendations)

# ==================== Função principal ====================
def gerar_recomendacao_completa(
        interacoes_usuarios_artistas,
        dados_artistas,
        caracteristicas_por_artistadata_by_artist,
        musicas_base,
        musical_features,
        preference_map,
        generos_mapeamento,
        generos_indesejados_map=None,
        num_recommendations_needed=8,
        pesos_atuais=None
):
    def padronizar_genero(g):
        return g.lower().replace("-", " ").strip()

    dfs_cols = [
        (dados_artistas, 'name', 'name_clean'),
        (caracteristicas_por_artistadata_by_artist, 'artists', 'artists_clean'),
        (musicas_base, 'artists', 'artists_clean'),
    ]
    for df_, col_in, col_out in dfs_cols:
        df_[col_out] = (
            df_[col_in]
            .astype(str)
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    musicas_base['genres_lower'] = musicas_base['artist_genres'].apply(
        lambda gs: [g.lower() for g in gs] if isinstance(gs, list) else [])

    # === SIMULAÇÃO DE PERFIS DE USUÁRIOS PARA CLUSTERIZAÇÃO ===
    num_perfis_simulados = 5000
    user_profiles = gerar_perfis_aleatorios(num_perfis_simulados, musical_features, preference_map)

    # --- Pré-processamento e Clusterização com dados demográficos ---
    user_profiles_encoded = pd.get_dummies(user_profiles, columns=['gender', 'country'])

    features_para_clusterizar = musical_features + ['age']
    X_features_modelo = [f for f in user_profiles_encoded.columns if
                         f in features_para_clusterizar or 'gender' in f or 'country' in f]

    # Escalonamento dos dados
    scaler_modelo = StandardScaler()
    X_scaled_modelo = scaler_modelo.fit_transform(user_profiles_encoded[X_features_modelo])

    # PCA para redução de dimensionalidade
    n_components = min(len(X_features_modelo), 10)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled_modelo)

    k_ideal = encontrar_melhor_k(X_pca)
    kmeans = KMeans(n_clusters=k_ideal, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    print(f"KMeans executado. Encontrou {len(np.unique(labels))} clusters.")

    user_profiles_encoded['cluster'] = labels
    user_profiles_encoded = ajustar_clusters(user_profiles_encoded, min_size=10)

    print("Clusters finais:")
    print(user_profiles_encoded['cluster'].value_counts())

    # --- Usuário e persona ---
    novo_usuario_info = chamada_api_retry(gerar_novo_usuario_aleatorio)
    if novo_usuario_info:
        novo_usuario = novo_usuario_info
        persona_info = chamada_api_retry(criar_persona_gemini, novo_usuario, preference_map)
        if persona_info:
            persona_gerada = persona_info['persona_text']
            info_detalhada = persona_info['persona_info']
            is_fallback = False
        else:
            print("\nAVISO: Falha na API do Gemini para criar persona. Usando fallback offline.")
            is_fallback = True
    else:
        print("\nAVISO: Falha na API do Gemini para gerar usuário. Usando fallback offline.")
        is_fallback = True

    if is_fallback:
        novo_usuario = {
            'age': 30, 'gender': 'f', 'country': 'Brazil',
            'energy': 'alto', 'tempo': 'medio', 'acousticness': 'medio',
            'instrumentalness': 'baixo', 'speechiness': 'baixo', 'liveness': 'baixo'
        }
        persona_gerada = "Sou uma pessoa vibrante que gosta de música animada, com um bom ritmo, perfeita para dançar. Prefiro músicas com vocais e que não sejam muito acústicas."
        info_detalhada = {
            'generos': ['pop', 'dance'], 'subgeneros': ['synth-pop', 'electropop'],
            'artistas_relacionados': ['Dua Lipa', 'Lady Gaga', 'The Weeknd']
        }

    # Acessa o dicionário 'persona_info' para obter os dados detalhados
    persona_generos = info_detalhada.get('generos', [])
    persona_subgeneros = info_detalhada.get('subgeneros', [])
    persona_artistas = info_detalhada.get('artistas_relacionados', [])

    generos_para_filtro = mapear_generos_artista(persona_generos) + mapear_generos_artista(persona_subgeneros)
    persona_generos_mapeados = [padronizar_genero(g) for g in generos_para_filtro]
    print("Generos mapeados:", persona_generos_mapeados)

    # GARANTINDO QUE AS FEATURES DO NOVO USUÁRIO SÃO STRINGS DE PREFERÊNCIA
    for feat in musical_features:
        if feat not in novo_usuario:
            novo_usuario[feat] = np.random.choice(list(preference_map[feat].keys()))

    # === Lógica para definir pesos dinâmicos (AGORA MAIS INTELIGENTE) ===
    if pesos_atuais:
        pesos_dinamicos = pesos_atuais
    else:
        pesos_base = {
            'genero': 0.25, 'energy': 0.15, 'tempo': 0.15,
            'popularity': 0.10, 'instrumentalness': 0.15,
            'speechiness': 0.10, 'liveness': 0.10,
        }
        # Na seção 'Ajusta pesos com base nas preferências da persona'
        for feature in ['energy', 'tempo', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'liveness',
                        'speechiness']:
            pref = novo_usuario.get(feature, 'medio')
            if pref == 'alto':
                # Dobre o peso de uma característica que é muito alta para a persona
                if feature in pesos_base:
                    pesos_base[feature] *= 2.0
                else:
                    pesos_base[feature] = 0.20  # Se for uma nova feature
            elif pref == 'baixo':
                # Reduza drasticamente o peso de uma característica que é muito baixa
                if feature in pesos_base:
                    pesos_base[feature] *= 0.1

        speechiness_pref = novo_usuario.get('speechiness')
        instrumentalness_pref = novo_usuario.get('instrumentalness')
        if speechiness_pref == 'alto':
            pesos_base['speechiness'] += 0.08
            if instrumentalness_pref != 'alto':
                pesos_base['instrumentalness'] -= 0.05
        if instrumentalness_pref == 'alto':
            pesos_base['instrumentalness'] += 0.08
            if speechiness_pref != 'alto':
                pesos_base['speechiness'] -= 0.05

        soma_pesos = sum(pesos_base.values())
        pesos_dinamicos = {k: v / soma_pesos for k, v in pesos_base.items()}

    # --- Encontrar o cluster do novo usuário ---
    new_user_data_modelo = {
        'age': novo_usuario.get('age'),
        'gender': novo_usuario.get('gender'),
        'country': novo_usuario.get('country')
    }
    for feat in musical_features:
        val = novo_usuario.get(feat, None)
        if isinstance(val, str):
            new_user_data_modelo[feat] = preference_map[feat][val.lower()]
        else:
            new_user_data_modelo[feat] = user_profiles[feat].mean()

    new_user_df_modelo = pd.DataFrame([new_user_data_modelo])
    new_user_encoded = pd.get_dummies(new_user_df_modelo, columns=['gender', 'country'])

    for col in X_features_modelo:
        if col not in new_user_encoded.columns:
            new_user_encoded[col] = 0

    new_user_encoded = new_user_encoded[X_features_modelo]
    new_user_scaled = scaler_modelo.transform(new_user_encoded)
    new_user_pca = pca.transform(new_user_scaled)

    clusters_centroids = user_profiles_encoded.groupby('cluster')[X_features_modelo].mean()
    clusters_centroids_scaled = scaler_modelo.transform(clusters_centroids)
    clusters_centroids_pca = pca.transform(clusters_centroids_scaled)

    cluster_labels = user_profiles_encoded['cluster'].unique()
    cluster_labels.sort()

    dists = np.linalg.norm(clusters_centroids_pca - new_user_pca, axis=1).flatten()
    ordem = np.argsort(dists)

    total_clusters = len(cluster_labels)
    clusters_relevantes = [cluster_labels[i] for i in ordem]

    num_clusters_a_buscar = min(5, total_clusters)
    clusters_para_buscar = clusters_relevantes[:num_clusters_a_buscar]
    print(f"O novo usuário está relacionado aos clusters (top-{len(clusters_para_buscar)}): {clusters_para_buscar}")

    # === LÓGICA DE RECOMENDAÇÃO REFINADA ===

    # 1. Pontua todas as músicas da base de dados
    generos_indesejados = generos_indesejados_map.get(
        'default', []
    ) if generos_indesejados_map else []

    if 'rock' in persona_generos_mapeados:
        generos_indesejados.extend(generos_indesejados_map.get('rock_profile', []))
    elif 'hip hop' in persona_generos_mapeados:
        generos_indesejados.extend(generos_indesejados_map.get('hip_hop_profile', []))
    elif 'pop' in persona_generos_mapeados:
        generos_indesejados.extend(generos_indesejados_map.get('pop_profile', []))
    elif 'electronic' in persona_generos_mapeados:
        generos_indesejados.extend(generos_indesejados_map.get('electronic_profile', []))

    if 'indie' in persona_generos_mapeados or 'folk' in persona_generos_mapeados or 'acoustic' in persona_generos_mapeados:
        generos_indesejados.extend(generos_indesejados_map.get('indie_pop_folk_profile', []))

    persona_para_filtro = {
        'generos': persona_generos_mapeados,
        'subgeneros': persona_subgeneros,
        'energy': novo_usuario['energy'],
        'tempo': novo_usuario['tempo'],
        'instrumentalness': novo_usuario['instrumentalness'],
        'speechiness': novo_usuario['speechiness'],
        'liveness': novo_usuario['liveness']
    }

    musicas_pontuadas_base = filtrar_playlist(musicas_base.copy(), generos_mapeamento, persona_para_filtro,
                                              pesos_dinamicos, top_n=len(musicas_base),
                                              generos_indesejados=generos_indesejados)

    # 2. Obtém a lista de artistas dos clusters relevantes
    user_ids_clusters = user_profiles_encoded[user_profiles_encoded['cluster'].isin(clusters_para_buscar)][
        'userID'].tolist()

    interacoes_filtradas = interacoes_usuarios_artistas[interacoes_usuarios_artistas['userID'].isin(user_ids_clusters)]
    artist_ids_clusters = interacoes_filtradas['artistID'].unique()
    artists_from_clusters = dados_artistas[dados_artistas['id'].isin(artist_ids_clusters)]

    artists_clean_persona = [
        art.lower().replace(r'[^\w\s]', '').replace(r'\s+', ' ').strip()
        for art in persona_artistas
    ]

    artists_clean_from_clusters = artists_from_clusters['name_clean'].tolist()

    artists_to_search = list(set(artists_clean_from_clusters + artists_clean_persona))

    # 3. Seleciona as músicas mais bem pontuadas que pertencem aos clusters ou aos artistas da persona
    playlist_cluster = musicas_pontuadas_base[musicas_pontuadas_base['artists_clean'].isin(artists_to_search)].copy()

    if not playlist_cluster.empty and artists_clean_persona:
        playlist_cluster['is_persona_artist'] = playlist_cluster['artists_clean'].isin(artists_clean_persona)
        playlist_cluster = playlist_cluster.sort_values(by=['is_persona_artist', 'final_score'],
                                                        ascending=[False, False])

    print(f"INFO: {len(playlist_cluster)} músicas encontradas nos clusters e/ou artistas da persona.")

    # APLICAÇÃO DO BALANCEAMENTO DE POPULARIDADE E DIVERSIDADE
    if not playlist_cluster.empty:
        # Ponderação de popularidade
        pesos_popularidade = {'top': 0.4, 'mid': 0.4, 'low': 0.2}
        playlist_filtrada = balancear_popularidade(playlist_cluster, num_recommendations_needed, pesos_popularidade)

        # Lógica de diversidade (limites de artista/gênero)
        playlist_final = selecionar_musicas_diversas(
            playlist_filtrada, num_recommendations_needed,
            artist_cap=3,  # no máximo 1 música por artista
            genre_cap=4,  # no máximo 4 músicas por gênero
            min_genres=3  # garante pelo menos 3 gêneros na playlist
        )
    else:
        print("\n=== NENHUMA MÚSICA SELECIONADA DOS CLUSTERS RELEVANTES. USANDO A LÓGICA DE FALLBACK ===")
        playlist_final = pd.DataFrame()  # Cria um DataFrame vazio para o fallback

    # 4. Fallback final se a playlist ainda não estiver cheia
    if len(playlist_final) < num_recommendations_needed:
        faltantes = num_recommendations_needed - len(playlist_final)
        print(
            f"\nINFO: Apenas {len(playlist_final)} músicas foram geradas na busca primária. Buscando mais {faltantes} no fallback inteligente.")

        existing_ids = set(playlist_final['id'].unique()) if not playlist_final.empty else set()

        # Passo 1 do Fallback: Buscar músicas dos artistas da persona que não foram selecionadas ainda
        musicas_artistas_persona = musicas_pontuadas_base[
            (musicas_pontuadas_base['artists_clean'].isin(artists_clean_persona)) &
            (~musicas_pontuadas_base['id'].isin(existing_ids))
            ].sort_values(by='final_score', ascending=False).copy()

        if len(musicas_artistas_persona) > 0:
            playlist_final = pd.concat([playlist_final, musicas_artistas_persona.head(faltantes)], ignore_index=True)
            print(
                f"INFO: Fallback 1 ativado: {len(musicas_artistas_persona.head(faltantes))} músicas de artistas da persona foram adicionadas.")
            faltantes = num_recommendations_needed - len(playlist_final)
            existing_ids.update(playlist_final['id'].unique())

        # Passo 2 do Fallback: Buscar músicas com correspondência de gênero
        if faltantes > 0:
            musicas_genero_match = musicas_pontuadas_base[
                (musicas_pontuadas_base['genero_score_norm'] > 0) &
                (~musicas_pontuadas_base['id'].isin(existing_ids))
                ].copy()

            if len(musicas_genero_match) > 0:
                musicas_genero_match_head = musicas_genero_match.head(faltantes)
                playlist_final = pd.concat([playlist_final, musicas_genero_match_head], ignore_index=True)
                print(
                    f"INFO: Fallback 2 ativado: {len(musicas_genero_match_head)} músicas com gênero correspondente adicionadas.")
                faltantes = num_recommendations_needed - len(playlist_final)
                existing_ids.update(playlist_final['id'].unique())

        # Passo 3 do Fallback: Fallback padrão (melhores músicas restantes)
        if faltantes > 0:
            musicas_adicionais = musicas_pontuadas_base[~musicas_pontuadas_base['id'].isin(existing_ids)].copy()
            musicas_adicionais_head = musicas_adicionais.head(faltantes)
            playlist_final = pd.concat([playlist_final, musicas_adicionais_head], ignore_index=True)
            print(f"INFO: Fallback 3 ativado: {len(musicas_adicionais_head)} músicas genéricas adicionadas.")

    cols_necessarias = ['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features
    for c in cols_necessarias:
        if c not in playlist_final.columns:
            playlist_final[c] = None
    playlist_final = playlist_final[cols_necessarias]

    if not playlist_final.empty:
        nomes_musicas = [f"{row['name']} - {row['artists']}" for _, row in playlist_final.iterrows()]
        reacao = chamada_api_retry(obter_reacao_persona, nomes_musicas, persona_gerada, novo_usuario)
        if not reacao:
            print("\nAVISO: Falha na API do Gemini para obter reação. Usando reação padrão.")
            reacao = "🎧 Playlist gerada, mas a reação da persona não pôde ser obtida. Parece uma boa seleção!"
    else:
        reacao = "🎧 Nenhuma música recomendada."

    return novo_usuario, persona_gerada, playlist_final, reacao, pesos_dinamicos


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
    'tempo': {'alto': 190.0, 'medio': 110.0, 'baixo': 50.0},
}

generos_mapeamento = {
    'rock': ['rock', 'classic rock', 'hard rock', 'rock and roll', 'alternative rock'],
    'pop': ['pop', 'indie pop', 'pop rock', 'synth-pop', 'electropop'],
    'electronic': ['edm', 'dance pop', 'electronic', 'house', 'dubstep', 'techno', 'trance', 'electronica',
                   'electro house', 'tech house', 'progressive house'],
    'hip hop': ['hip hop', 'rap', 'trap', 'drill'],
    'r&b': ['r&b', 'soul'],
    'jazz': ['jazz', 'smooth jazz', 'swing'],
    'folk': ['folk', 'singer-songwriter'],
    'metal': ['metal', 'heavy metal', 'death metal', 'black metal', 'thrash metal'],
    'classical': ['classical', 'symphony', 'orchestral'],
    'blues': ['blues', 'delta blues', 'electric blues'],
    'indie': ['indie pop', 'indie rock', 'indie folk'],
    'country': ['country', 'alt-country'],
    'dance': ['dance pop', 'edm', 'house', 'tech house', 'progressive house', 'electro house', 'electronic'],
    'disco': ['disco', 'funk'],
    'reggae': ['reggae', 'dub'],
    'latin': ['latin', 'reggaeton', 'salsa', 'bachata'],
    'k-pop': ['k-pop'],
    'j-pop': ['j-pop'],
    'synthpop': ['synthpop', 'synth-pop', 'electropop', 'electro pop'],
    'alternative': ['alternative', 'alt rock', 'alt pop'],
}

generos_indesejados_map = {
    'default': ['sertanejo', 'pagode', 'samba', 'forro', 'sertanejo universitario', 'sertanejo pop'],
    'rock_profile': ['reggaeton', 'funk carioca', 'k-pop', 'latin', 'bachata', 'corrido'],
    'hip_hop_profile': ['country', 'sertanejo', 'symphony', 'classical', 'orchestral', 'opera'],
    'pop_profile': ['death metal', 'black metal', 'thrash metal', 'hardcore', 'grindcore'],
    'electronic_profile': ['heavy metal', 'black metal', 'thrash metal', 'hardcore', 'grindcore', 'country',
                           'sertanejo', 'samba', 'forro', 'reggaeton', 'bachata', 'corrido', 'funk carioca'],
    'indie_pop_folk_profile': ['k-pop', 'reggaeton', 'funk carioca', 'hip hop', 'rap', 'trap', 'drill', 'grindcore',
                               'death metal', 'black metal', 'thrash metal', 'sertanejo', 'pagode', 'samba', 'forro',
                               'sertanejo universitario', 'sertanejo pop', 'bachata', 'corrido']
}

try:
    musicas_base = pd.read_csv('./datasets/musicas_com_generos.csv')
    musicas_base['artist_genres'] = musicas_base['artist_genres'].apply(ast.literal_eval)
except FileNotFoundError:
    print("Erro: O arquivo 'musicas_com_generos.csv' não foi encontrado. Executando o pré-processamento...")
    unificar_e_salvar_generos()
    musicas_base = pd.read_csv('./datasets/musicas_com_generos.csv')
    musicas_base['artist_genres'] = musicas_base['artist_genres'].apply(ast.literal_eval)

if musicas_base is not None:
    # Simula o loop de feedback
    pesos_atuais = None  # Começa com pesos padrão
    for i in range(1):
        print("\n--- Execução " + str(i + 1) + " ---")

        # Gera a recomendação com os pesos da iteração anterior
        novo_usuario, persona, playlist, reacao, pesos_dinamicos_usados = gerar_recomendacao_completa(
            interacoes_usuarios_artistas=pd.read_csv('./datasets/user_artists.csv', sep='\t'),
            dados_artistas=pd.read_csv('./datasets/artists.csv', sep='\t'),
            caracteristicas_por_artistadata_by_artist=pd.read_csv('./datasets/data_by_artist.csv'),
            musicas_base=musicas_base,
            musical_features=musical_features,
            preference_map=preference_map,
            generos_mapeamento=generos_mapeamento,
            generos_indesejados_map=generos_indesejados_map,
            num_recommendations_needed=8,
            pesos_atuais=pesos_atuais
        )

        print("\n--- RESULTADO FINAL ---")
        print("Novo usuário:", novo_usuario)
        print("Persona gerada:", persona)
        print_playlist_bonita(playlist)

        if reacao:
            print("\n📝 Reação da persona:")
            print(reacao)
else:
    print("Erro: A base de dados 'musicas_com_generos.csv' não foi carregada.")