import pandas as pd
import numpy as np
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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


def filtrar_playlist(df_musicas, generos_mapeamento, persona, top_n=20):
    """
    Filtra a playlist de acordo com a persona:
    - Prioriza gêneros que batem com a persona
    - Ajusta por energy e tempo
    - Adiciona pontuação por popularidade
    - Mantém fallback caso não haja músicas suficientes nos clusters
    """
    # Padronizar gêneros do dataset
    df_musicas['genres_lower'] = df_musicas['artist_genres'].apply(
        lambda gs: [g.lower() for g in gs] if isinstance(gs, list) else [])

    # ======================
    # 1. Score de popularidade
    # ======================
    df_musicas['popularity_score'] = df_musicas['popularity'] / 100

    # ======================
    # 2. Score de correspondência de gênero
    # ======================
    def genero_score(musica_genres):
        score = 0
        generos_mapeados_musica = [g.lower() for g in mapear_generos_artista(musica_genres)]
        for g_persona in persona['genres']:
            # Verifica se o gênero da persona está na música, com uma correspondência próxima
            matches = difflib.get_close_matches(g_persona, generos_mapeados_musica, n=1, cutoff=0.7)
            if matches:
                # Dá um boost para o gênero encontrado
                score += 1
        return score

    df_musicas['genero_score'] = df_musicas['artist_genres'].apply(genero_score)
    df_musicas['genero_score_norm'] = df_musicas['genero_score'] / len(persona['genres'])

    # ======================
    # 3. Score de energia/tempo e INSTRUMENTAL
    # ======================
    energia_map = {'baixo': 0.2, 'medio': 0.5, 'alto': 0.8}
    tempo_map = {'baixo': 60, 'medio': 100, 'alto': 140}
    instrumental_map = {'baixo': 0.05, 'medio': 0.4, 'alto': 0.95}
    speechiness_map = {'baixo': 0.05, 'medio': 0.25, 'alto': 0.9}

    energy_persona = energia_map.get(persona['energy'], 0.5)
    tempo_persona = tempo_map.get(persona['tempo'], 110)
    instrumental_persona = instrumental_map.get(persona['instrumentalness'], 0.4)
    speechiness_persona = speechiness_map.get(persona['speechiness'], 0.25)

    df_musicas['energy_score'] = 1 - abs(
        df_musicas['energy'] - energy_persona) / 0.8
    df_musicas['tempo_score'] = 1 - abs(
        df_musicas['tempo'] - tempo_persona) / 140
    df_musicas['instrumentalness_score'] = 1 - abs(
        df_musicas['instrumentalness'] - instrumental_persona) / 0.95
    df_musicas['speechiness_score'] = 1 - abs(
        df_musicas['speechiness'] - speechiness_persona) / 0.9

    # ======================
    # 4. Score final (Pesos ajustados com base no CÓDIGO 6)
    # ======================
    df_musicas['final_score'] = (
            df_musicas['genero_score_norm'] * 0.15 +
            df_musicas['energy_score'] * 0.25 +
            df_musicas['tempo_score'] * 0.25 +
            df_musicas['popularity_score'] * 0.1 +
            df_musicas['instrumentalness_score'] * 0.15 +
            df_musicas['speechiness_score'] * 0.1
    )

    df_musicas_sorted = df_musicas.sort_values(by='final_score', ascending=False)
    df_musicas_sorted = df_musicas_sorted.drop_duplicates(subset=['artists'], keep='first')

    return df_musicas_sorted.head(top_n)


# ==================== Função para gerar playlist personalizada ====================
def gerar_playlist_personalizada(top_artists, clusters_relacionados, novo_usuario, X_features_playlist,
                                 qtd_por_cluster=2):
    """
    Gera playlist reordenada por proximidade do usuário a partir de 'top_artists'
    (um DataFrame com as colunas de features musicais e coluna 'cluster').
    """

    map_val = {"baixo": 0.2, "medio": 0.5, "alto": 0.8}
    persona_features = np.array([
        map_val[novo_usuario["danceability"]],
        map_val[novo_usuario["energy"]],
        map_val[novo_usuario["loudness"]],
        map_val[novo_usuario["valence"]],
        map_val[novo_usuario["tempo"]],
        map_val[novo_usuario["acousticness"]],
        map_val[novo_usuario["instrumentalness"]],
        map_val[novo_usuario["liveness"]],
        map_val[novo_usuario["speechiness"]],
    ]).reshape(1, -1)

    playlist_final = []
    for cluster in clusters_relacionados:
        candidatos = top_artists[top_artists['cluster'] == cluster].copy()
        if not candidatos.empty:
            X_candidatos = candidatos[X_features_playlist].values
            distancias = euclidean_distances(X_candidatos, persona_features).flatten()
            candidatos['distancia'] = distancias
            candidatos = candidatos.sort_values('distancia').drop_duplicates('name_clean').head(qtd_por_cluster)
            playlist_final.append(candidatos)

    if playlist_final:
        playlist_final = pd.concat(playlist_final, ignore_index=True).drop_duplicates('name_clean')
    else:
        playlist_final = pd.DataFrame()

    return playlist_final


# ==================== Função para encontrar o K ideal para KMeans ====================
def encontrar_melhor_k(X, k_range=(32, 50)):
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


# ==================== Função principal ====================
def gerar_recomendacao_completa(
        interacoes_usuarios_artistas,
        dados_artistas,
        caracteristicas_por_artistadata_by_artist,
        musicas_base,
        musical_features,
        demographic_features,
        preference_map,
        generos_mapeamento,
        num_recommendations_needed=10
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

    # --- CRIA PERFIS DE USUÁRIO MÉDIOS PARA CLUSTERING ---
    merged_with_features = pd.merge(interacoes_usuarios_artistas, dados_artistas,
                                    left_on='artistID', right_on='id', how='left')
    merged_with_features = pd.merge(merged_with_features,
                                    caracteristicas_por_artistadata_by_artist,
                                    left_on='name_clean', right_on='artists_clean', how='left')

    merged_with_features.dropna(subset=musical_features, inplace=True)

    # Simulação de dados demográficos
    unique_users = merged_with_features['userID'].unique()
    fake_users = pd.DataFrame({
        'userID': unique_users,
        'age': np.random.randint(15, 60, len(unique_users)),
        'gender': np.random.choice(['m', 'f'], len(unique_users)),
        'country': np.random.choice(['Brazil', 'USA', 'Germany', 'UK', 'Japan'], len(unique_users)),
    })

    # Adiciona os dados demográficos ao DataFrame principal
    merged_with_features = pd.merge(merged_with_features, fake_users, on='userID', how='left')

    # === SIMULAÇÃO DE PERFIS DE USUÁRIOS PARA CLUSTERIZAÇÃO ===
    # Geramos 5000 perfis fictícios para melhorar a robustez do clustering
    # Você pode alterar este número para testar o impacto na recomendação
    num_perfis_simulados = 5000
    user_profiles = gerar_perfis_aleatorios(num_perfis_simulados, musical_features, preference_map)

    # --- Pré-processamento e Clusterização com dados demográficos ---
    # Codificação de colunas categóricas
    user_profiles_encoded = pd.get_dummies(user_profiles, columns=['gender', 'country'])

    # Adiciona a feature 'age' e 'musical_features'
    features_para_clusterizar = musical_features + ['age']

    # As features para o modelo agora incluem as features musicais e as demográficas codificadas
    X_features_modelo = [f for f in user_profiles_encoded.columns if
                         f in features_para_clusterizar or 'gender' in f or 'country' in f]

    # Escalonamento dos dados
    scaler_modelo = StandardScaler()
    X_scaled_modelo = scaler_modelo.fit_transform(user_profiles_encoded[X_features_modelo])

    k_ideal = encontrar_melhor_k(X_scaled_modelo)
    kmeans = KMeans(n_clusters=k_ideal, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled_modelo)
    print(f"KMeans executado. Encontrou {len(np.unique(labels))} clusters.")

    user_profiles_encoded['cluster'] = labels
    user_profiles_encoded = ajustar_clusters(user_profiles_encoded, min_size=10)

    print("Clusters finais:")
    print(user_profiles_encoded['cluster'].value_counts())

    # --- Usuário e persona ---
    novo_usuario = chamada_api_retry(gerar_novo_usuario_aleatorio)
    persona_info = chamada_api_retry(criar_persona_gemini, novo_usuario, preference_map)
    persona_gerada = persona_info['persona_text']
    persona_generos = persona_info['persona_genres']

    generos_para_filtro = mapear_generos_artista(persona_generos)
    persona_generos_mapeados = [padronizar_genero(g) for g in generos_para_filtro]
    print("Generos mapeados:", persona_generos_mapeados)

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

    # Cria um DataFrame para o novo usuário com todas as colunas
    new_user_df_modelo = pd.DataFrame([new_user_data_modelo])

    # Codifica o novo usuário da mesma forma que o dataset
    new_user_encoded = pd.get_dummies(new_user_df_modelo, columns=['gender', 'country'])

    # Garante que as colunas do novo usuário e do dataset são as mesmas
    for col in X_features_modelo:
        if col not in new_user_encoded.columns:
            new_user_encoded[col] = 0

    # Reordena as colunas para que correspondam às do dataset de treinamento
    new_user_encoded = new_user_encoded[X_features_modelo]

    scaler_user = StandardScaler()
    scaler_user.fit(user_profiles_encoded[X_features_modelo])
    new_user_scaled = scaler_user.transform(new_user_encoded)

    clusters_centroids = user_profiles_encoded.groupby('cluster')[X_features_modelo].mean().values
    cluster_labels = user_profiles_encoded['cluster'].unique()
    cluster_labels.sort()

    dists = np.linalg.norm(clusters_centroids - new_user_scaled, axis=1).flatten()
    ordem = np.argsort(dists)

    # === LÓGICA DINÂMICA PARA top_k_clusters ===
    # Calcula o número de clusters a serem considerados com base na quantidade
    # de músicas a serem recomendadas (num_recommendations_needed)
    # e na quantidade de músicas por cluster (assumida como 2, ou ajustável).
    musicas_por_cluster = 2
    total_clusters = len(cluster_labels)

    # 1. Tenta pegar a quantidade de clusters necessária para gerar o número de músicas desejado.
    clusters_necessarios = int(np.ceil(num_recommendations_needed / musicas_por_cluster))

    # 2. Garante que o valor não seja maior que a metade dos clusters,
    # para manter a relevância.
    top_k_clusters = min(clusters_necessarios, total_clusters // 2, 8)  # Limite o máximo a 8, por exemplo

    # 3. O valor mínimo deve ser 1 ou 2 clusters
    top_k_clusters = max(2, top_k_clusters)

    clusters_relacionados = [cluster_labels[i] for i in ordem[:top_k_clusters]]
    print(f"O novo usuário está relacionado aos clusters (top-{len(clusters_relacionados)}): {clusters_relacionados}")

    # === LÓGICA DE RECOMENDAÇÃO MELHORADA ===
    perfis_clusters_relacionados = user_profiles_encoded[
        user_profiles_encoded['cluster'].isin(clusters_relacionados)].copy()

    if not perfis_clusters_relacionados.empty:
        distancias_ao_novo_usuario = np.linalg.norm(
            perfis_clusters_relacionados[musical_features].values - new_user_scaled[:, :len(musical_features)], axis=1)

        pesos = 1 / (distancias_ao_novo_usuario + 1e-6)
        perfil_medio_grupos = np.average(perfis_clusters_relacionados[musical_features].values, axis=0, weights=pesos)
        perfil_medio_grupos = {feat: perfil_medio_grupos[i] for i, feat in enumerate(musical_features)}
    else:
        perfil_medio_grupos = {feat: user_profiles[feat].mean() for feat in musical_features}

    user_ids_clusters = user_profiles_encoded[user_profiles_encoded['cluster'].isin(clusters_relacionados)][
        'userID'].tolist()
    interacoes_filtradas = interacoes_usuarios_artistas[interacoes_usuarios_artistas['userID'].isin(user_ids_clusters)]
    artist_ids_clusters = interacoes_filtradas['artistID'].unique()
    artists_from_clusters = dados_artistas[dados_artistas['id'].isin(artist_ids_clusters)]

    # Agora, em vez de filtrar estritamente por gênero, passamos todas as músicas para o filtro.
    # A função 'filtrar_playlist' se encarregará de ranqueá-las
    # por gênero, energia e tempo, garantindo que as melhores sejam escolhidas.

    musicas_para_ordenar = musicas_base[musicas_base['artists_clean'].isin(artists_from_clusters['name_clean'])].copy()

    if musicas_para_ordenar.empty:
        print("INFO: Não há músicas de artistas dos clusters. Usando músicas da base completa.")
        musicas_para_ordenar = musicas_base.copy()

    perfil_clusters_vector = np.array([perfil_medio_grupos[feat] for feat in musical_features])
    for feat in musical_features:
        if feat not in musicas_para_ordenar.columns or musicas_para_ordenar[feat].isnull().all():
            musicas_para_ordenar[feat] = user_profiles[feat].mean()

    musicas_para_ordenar['distancia_perfil_cluster'] = np.linalg.norm(
        musicas_para_ordenar[musical_features].values - perfil_clusters_vector, axis=1
    )

    persona_para_filtro = {
        'genres': persona_generos_mapeados,
        'energy': novo_usuario['energy'],
        'tempo': novo_usuario['tempo'],
        'instrumentalness': novo_usuario['instrumentalness'],
        'speechiness': novo_usuario['speechiness']
    }

    playlist_final = filtrar_playlist(musicas_para_ordenar, generos_mapeamento, persona_para_filtro,
                                      top_n=num_recommendations_needed)

    if len(playlist_final) < num_recommendations_needed:
        faltantes = num_recommendations_needed - len(playlist_final)
        print(
            f"INFO: Apenas {len(playlist_final)} músicas foram geradas dos clusters. Buscando mais {faltantes} na base completa.")

        # Lógica de fallback melhorada: busca na base completa com o mesmo critério
        musicas_adicionais = filtrar_playlist(musicas_base.copy(), generos_mapeamento, persona_para_filtro,
                                              top_n=faltantes * 2)
        existing_ids = set(playlist_final['id'].unique())
        musicas_adicionais = musicas_adicionais[~musicas_adicionais['id'].isin(existing_ids)]
        playlist_final = pd.concat([playlist_final, musicas_adicionais.head(faltantes)], ignore_index=True)

    print(playlist_final[['name', 'artists', 'genero_score', 'energy_score', 'tempo_score',
                          'popularity_score', 'instrumentalness_score', 'speechiness_score', 'final_score']])

    if 'name' not in playlist_final.columns:
        playlist_final = pd.DataFrame(
            columns=['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features)

    contador_artistas = {}
    playlist_final_limited = []
    for _, row in playlist_final.iterrows():
        artista = row.get('artists', '')
        if contador_artistas.get(artista, 0) < 3:
            playlist_final_limited.append(row.to_dict())
            contador_artistas[artista] = contador_artistas.get(artista, 0) + 1
        if len(playlist_final_limited) >= num_recommendations_needed:
            break

    playlist_final = pd.DataFrame(playlist_final_limited)

    if len(playlist_final) < num_recommendations_needed:
        faltantes = num_recommendations_needed - len(playlist_final)
        # Fallback inteligente: buscar as melhores na base toda, não apenas amostras aleatórias
        musicas_adicionais_fallback = filtrar_playlist(musicas_base.copy(), generos_mapeamento, persona_para_filtro,
                                                       top_n=faltantes * 2)
        existing_ids = set(playlist_final['id'].unique())
        musicas_adicionais_fallback = musicas_adicionais_fallback[~musicas_adicionais_fallback['id'].isin(existing_ids)]
        playlist_final = pd.concat([playlist_final, musicas_adicionais_fallback.head(faltantes)], ignore_index=True)
        print("INFO: Gerando músicas adicionais com a lógica de fallback inteligente para completar a playlist.")

    cols_necessarias = ['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features
    for c in cols_necessarias:
        if c not in playlist_final.columns:
            playlist_final[c] = None
    playlist_final = playlist_final[cols_necessarias]

    if not playlist_final.empty:
        nomes_musicas = [f"{row['name']} - {row['artists']}" for _, row in playlist_final.iterrows()]
        reacao = chamada_api_retry(obter_reacao_persona, nomes_musicas, persona_gerada, novo_usuario)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    return novo_usuario, persona_gerada, playlist_final, reacao


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

demographic_features = ['age', 'gender', 'country']

try:
    musicas_base = pd.read_csv('./datasets/musicas_com_generos.csv')
    musicas_base['artist_genres'] = musicas_base['artist_genres'].apply(ast.literal_eval)
except FileNotFoundError:
    print("Erro: O arquivo 'musicas_com_generos.csv' não foi encontrado. Executando o pré-processamento...")
    unificar_e_salvar_generos()
    musicas_base = pd.read_csv('./datasets/musicas_com_generos.csv')
    musicas_base['artist_genres'] = musicas_base['artist_genres'].apply(ast.literal_eval)

if musicas_base is not None:
    novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa(
        interacoes_usuarios_artistas=pd.read_csv('./datasets/user_artists.csv', sep='\t'),
        dados_artistas=pd.read_csv('./datasets/artists.csv', sep='\t'),
        caracteristicas_por_artistadata_by_artist=pd.read_csv('./datasets/data_by_artist.csv'),
        musicas_base=musicas_base,
        musical_features=musical_features,
        demographic_features=demographic_features,
        preference_map=preference_map,
        generos_mapeamento=generos_mapeamento,
        num_recommendations_needed=8
    )

    print("Novo usuário:", novo_usuario)
    print("Persona gerada:", persona)
    print_playlist_bonita(playlist)
    if reacao:
        print("\n📝 Reação da persona:")
        print(reacao)
else:
    print("Erro: A base de dados 'musicas_com_generos.csv' não foi carregada.")