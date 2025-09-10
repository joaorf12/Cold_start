import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from preprocessamento import criar_df_generos_unificado
from visualizacao import plotar_clusters, print_playlist_bonita
from mapeamento_generos import mapear_generos_artista
from chamadasGemini import (
    chamada_api_retry,
    gerar_novo_usuario_aleatorio,
    criar_persona_gemini,
    obter_reacao_persona,
)
import warnings

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
def ajustar_clusters(df, X_features, min_size=5):
    clusters = df['cluster'].unique()
    if len(clusters) <= 1:
        return df

    centroids = df.groupby('cluster')[X_features].mean()
    dist_matrix = euclidean_distances(centroids, centroids)

    for c in clusters:
        mask_c = df['cluster'] == c
        if mask_c.sum() < min_size:
            # Pega o cluster mais próximo, excluindo ele mesmo
            vizinho_idx = np.argsort(dist_matrix[c])[1]
            vizinho = clusters[vizinho_idx]
            df.loc[mask_c, 'cluster'] = vizinho
    return df


def filtrar_playlist(df_musicas, generos_mapeamento, persona, top_n=20):
    """
    Filtra a playlist de acordo com a persona:
    - Prioriza gêneros que batem com a persona
    - Ajusta por energy e tempo
    - Mantém fallback caso não haja músicas suficientes nos clusters
    """
    # ... (Resto da função, antes da definição do genero_score) ...
    # Dicionário de gêneros negativos. Ajuste essa lista se necessário
    generos_negativos = generos_mapeamento

    # Padronizar gêneros do dataset
    df_musicas['genres_lower'] = df_musicas['artist_genres'].apply(lambda gs: [g.lower() for g in gs])

    # ======================
    # 2. Score de correspondência de gênero
    # ======================
    def genero_score(musica_genres):
        # Mapear os gêneros da música antes de comparar
        generos_mapeados_musica = mapear_generos_artista(musica_genres)

        # Aumente a pontuação de gênero para dar mais peso
        score = len(set(generos_mapeados_musica) & set(persona['genres']))

        # --- NOVA LÓGICA: PENALIDADE POR GÊNERO CONFLITANTE ---
        if score > 0:
            for genero_persona_principal in persona['genres']:
                genero_persona_principal_lower = genero_persona_principal.lower()
                if genero_persona_principal_lower in generos_negativos:
                    for genero_negativo in generos_negativos[genero_persona_principal_lower]:
                        if genero_negativo in generos_mapeados_musica:
                            score -= 1  # Aplica penalidade
                            break  # Pula para o próximo gênero
        return score

    df_musicas['genero_score'] = df_musicas['genres_lower'].apply(genero_score)

    # ======================
    # 3. Score de energia/tempo
    # ======================
    energia_map = {'baixo': 0.2, 'medio': 0.5, 'alto': 0.8}
    tempo_map = {'baixo': 60, 'medio': 100, 'alto': 140}

    energy_persona = energia_map[persona['energy']]
    tempo_persona = tempo_map[persona['tempo']]

    # Normalizar entre 0 e 1
    # Use a distância absoluta para um cálculo mais direto de proximidade
    df_musicas['energy_score'] = 1 - abs(
        df_musicas['energy'] - energy_persona) / 0.8  # Normalizado por um valor máximo de diferença
    df_musicas['tempo_score'] = 1 - abs(
        df_musicas['tempo'] - tempo_persona) / 140  # Normalizado por um valor máximo de diferença

    # ======================
    # 4. Score final - AJUSTE AQUI
    # ======================
    # Mantenha os pesos ajustados ou experimente outros
    df_musicas['final_score'] = (
            df_musicas['genero_score'] * 0.5 +
            df_musicas['energy_score'] * 0.3 +
            df_musicas['tempo_score'] * 0.2
    )

    # Filtra músicas com 'genero_score' > 0 para garantir que a música tenha pelo menos 1 gênero correspondente
    # Este é um filtro essencial para remover lixo
    df_musicas = df_musicas[df_musicas['genero_score'] > 0]

    # ======================
    # 5. Ordenar e pegar top N
    # ======================
    df_musicas_sorted = df_musicas.sort_values(by='final_score', ascending=False)

    # Opcional: Remova duplicatas de artista para maior diversidade
    df_musicas_sorted = df_musicas_sorted.drop_duplicates(subset=['artists'], keep='first')

    return df_musicas_sorted.head(top_n)

# ==================== Função para gerar playlist personalizada ====================
def gerar_playlist_personalizada(top_artists, clusters_relacionados, novo_usuario, X_features_playlist,
                                 qtd_por_cluster=2):
    """
    Gera playlist reordenada por proximidade do usuário a partir de 'top_artists'
    (um DataFrame com as colunas de features musicais e coluna 'cluster').
    """

    # === Mapeamento dos atributos qualitativos para valores numéricos ===
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

    # === Seleção de músicas mais próximas por cluster ===
    playlist_final = []
    for cluster in clusters_relacionados:
        candidatos = top_artists[top_artists['cluster'] == cluster].copy()
        if not candidatos.empty:
            X_candidatos = candidatos[X_features_playlist].values
            distancias = euclidean_distances(X_candidatos, persona_features).flatten()
            candidatos['distancia'] = distancias
            # Pega os N mais próximos e evita repetir artistas/músicas pelo 'name_clean'
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

        # Silhouette Score para avaliar a qualidade
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
        musicas_com_generos,
        musical_features,
        label_cols,
        preference_map,
        generos_mapeamento,
        num_recommendations_needed=10,
        top_k_clusters=6,
):

    def obter_generos_mapeados(generos_persona):
        generos_traduzidos = set()
        for genero_persona in generos_persona:
            genero_lower = genero_persona.lower()
            for chave, valores in generos_mapeamento.items():
                if chave in genero_lower:
                    generos_traduzidos.update(valores)
        return list(generos_traduzidos)

    def padronizar_genero(g):
        return g.lower().replace("-", " ").strip()

    # --- Limpeza e padronização de nomes ---
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

    # Merge de músicas com gêneros (usa DF já unificado vindo do preprocessamento)
    data_with_genres = pd.merge(musicas_base, musicas_com_generos, on='artists_clean', how='left')
    data_with_genres.rename(columns={'artists_x': 'artists', 'genres': 'artist_genres'}, inplace=True)
    data_with_genres.drop(columns=['artists_y'], inplace=True, errors='ignore')
    # Garante lista
    data_with_genres['artist_genres'] = data_with_genres['artist_genres'].apply(
        lambda x: x if isinstance(x, list) else (x.split(',') if isinstance(x, str) and x else [])
    )

    # --- NOVA LÓGICA: CRIA PERFIS DE USUÁRIO MÉDIOS PARA CLUSTERING ---
    merged_with_features = pd.merge(interacoes_usuarios_artistas, dados_artistas,
                                    left_on='artistID', right_on='id', how='left')
    merged_with_features = pd.merge(merged_with_features,
                                    caracteristicas_por_artistadata_by_artist,
                                    left_on='name_clean', right_on='artists_clean', how='left')

    # Garante que temos as características musicais e apenas usuários com dados
    merged_with_features.dropna(subset=musical_features, inplace=True)

    # Média ponderada por 'weight' para criar perfis de usuário mais precisos
    user_profiles = pd.DataFrame()
    for feature in musical_features:
        user_profiles[feature] = merged_with_features.groupby('userID').apply(
            lambda x: np.average(x[feature], weights=x['weight'])
        )

    user_profiles = user_profiles.reset_index()

    # Preencher NaNs remanescentes com a média geral das features
    user_profiles[musical_features] = user_profiles[musical_features].fillna(user_profiles[musical_features].mean())

    # O clustering agora será em 'user_profiles'
    X_features_modelo = musical_features  # Remove as features demográficas
    scaler_modelo = StandardScaler()
    X_scaled_modelo = scaler_modelo.fit_transform(user_profiles[X_features_modelo])

    # =========================
    # KMeans como única opção
    # =========================
    k_ideal = encontrar_melhor_k(X_scaled_modelo)
    kmeans = KMeans(n_clusters=k_ideal, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled_modelo)
    print(f"KMeans executado. Encontrou {len(np.unique(labels))} clusters.")

    user_profiles['cluster'] = labels

    # Ajusta clusters pequenos (para KMeans)
    user_profiles = ajustar_clusters(user_profiles, X_features_modelo, min_size=5)

    print("Clusters finais:")
    print(user_profiles['cluster'].value_counts())

    # --- Usuário e persona (única vez, coerente com preference_map) ---
    novo_usuario = chamada_api_retry(gerar_novo_usuario_aleatorio)
    persona_info = chamada_api_retry(criar_persona_gemini, novo_usuario, preference_map)
    persona_gerada = persona_info['persona_text']
    persona_generos = persona_info['persona_genres']

    # Item 2 - Mapear gêneros da persona para os gêneros genéricos do seu dicionário
    generos_para_filtro = mapear_generos_artista(persona_generos)
    persona_generos_mapeados = [padronizar_genero(g) for g in generos_para_filtro]

    print("Generos mapeados:", persona_generos_mapeados)

    # --- Encontrar o cluster do novo usuário ---
    new_user_data_modelo = {}
    for feat in musical_features:
        val = novo_usuario.get(feat, None)
        if isinstance(val, str):
            new_user_data_modelo[feat] = preference_map[feat][val.lower()]
        else:
            new_user_data_modelo[feat] = user_profiles[feat].mean()

    new_user_df_modelo = pd.DataFrame([new_user_data_modelo])[X_features_modelo]
    scaler_user = StandardScaler().fit(user_profiles[X_features_modelo])
    new_user_scaled = scaler_user.transform(new_user_df_modelo)

    # Distância do novo usuário aos centróides
    clusters_centroids = user_profiles.groupby('cluster')[X_features_modelo].mean().values

    # Mapeia os índices de volta para os rótulos de cluster
    cluster_labels = user_profiles['cluster'].unique()
    cluster_labels.sort()

    dists = np.linalg.norm(clusters_centroids - new_user_scaled, axis=1).flatten()
    ordem = np.argsort(dists)
    clusters_relacionados = [cluster_labels[i] for i in ordem[:min(top_k_clusters, len(cluster_labels))]]
    print(f"O novo usuário está relacionado aos clusters (top-{len(clusters_relacionados)}): {clusters_relacionados}")

    # === LÓGICA DE RECOMENDAÇÃO MELHORADA ===

    # 1. Pega os perfis médios dos clusters relacionados
    perfis_clusters_relacionados = user_profiles[user_profiles['cluster'].isin(clusters_relacionados)].copy()

    if not perfis_clusters_relacionados.empty:
        # 2. Calcula a distância de cada perfil de usuário dentro do cluster até o novo usuário
        distancias_ao_novo_usuario = np.linalg.norm(
            perfis_clusters_relacionados[musical_features].values - new_user_scaled, axis=1)

        # 3. Inverte a distância para criar um peso (quanto menor a distância, maior o peso)
        pesos = 1 / (distancias_ao_novo_usuario + 1e-6)

        # 4. Calcula a média ponderada para criar um perfil de ranqueamento
        perfil_medio_grupos = np.average(perfis_clusters_relacionados[musical_features].values, axis=0, weights=pesos)
        perfil_medio_grupos = {feat: perfil_medio_grupos[i] for i, feat in enumerate(musical_features)}
    else:
        # Fallback caso os clusters estejam vazios
        perfil_medio_grupos = {feat: user_profiles[feat].mean() for feat in musical_features}

    # --- Ponto de alteração: filtrar músicas APENAS dos clusters relacionados ---

    # Obtém as IDs de usuário dos clusters relacionados
    user_ids_clusters = user_profiles[user_profiles['cluster'].isin(clusters_relacionados)]['userID'].tolist()

    # Filtra as interações e depois as músicas
    interacoes_filtradas = interacoes_usuarios_artistas[interacoes_usuarios_artistas['userID'].isin(user_ids_clusters)]
    # Usar .unique() para pegar apenas os artistIDs distintos
    artist_ids_clusters = interacoes_filtradas['artistID'].unique()

    # Filtra os DataFrames de artistas e músicas
    artists_from_clusters = dados_artistas[dados_artistas['id'].isin(artist_ids_clusters)]

    # Garante que as músicas de 'data_with_genres' correspondem aos artistas
    musicas_filtradas = data_with_genres[
        data_with_genres['artists_clean'].isin(artists_from_clusters['name_clean'])].copy()

    # 3. Ranqueia as músicas FILTRADAS
    musicas_para_ordenar = musicas_filtradas

    # Ponto de entrada: o vetor de perfil médio dos clusters
    perfil_clusters_vector = np.array([perfil_medio_grupos[feat] for feat in musical_features])

    # Garante que as colunas musicais existem no dataframe
    for feat in musical_features:
        if feat not in musicas_para_ordenar.columns or musicas_para_ordenar[feat].isnull().all():
            musicas_para_ordenar[feat] = user_profiles[feat].mean()

    musicas_para_ordenar['distancia_perfil_cluster'] = np.linalg.norm(
        musicas_para_ordenar[musical_features].values - perfil_clusters_vector, axis=1
    )

    # 4. Aplica a lógica da persona para refinar a seleção
    persona_para_filtro = {
        'genres': persona_generos_mapeados,
        'energy': novo_usuario['energy'],
        'tempo': novo_usuario['tempo']
    }

    # Usa a função `filtrar_playlist` para ranquear as músicas
    playlist_final = filtrar_playlist(musicas_para_ordenar, generos_mapeamento, persona_para_filtro,
                                      top_n=num_recommendations_needed)

    # Se a playlist final estiver vazia após a filtragem, use a base inteira como fallback
    if playlist_final.empty:
        print(
            "INFO: Não foi possível gerar uma playlist com os clusters. Usando a base de dados completa como fallback.")
        playlist_final = filtrar_playlist(data_with_genres.copy(), generos_mapeamento, persona_para_filtro,
                                          top_n=num_recommendations_needed)

    print(playlist_final[['name', 'artists', 'genero_score', 'energy_score', 'tempo_score', 'final_score']])

    # --- Depois de calcular playlist_final ---
    if 'name' not in playlist_final.columns:
        playlist_final = pd.DataFrame(
            columns=['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features)

    # --- Limita repetição de artista (máx 3 vezes) ---
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

    # --- Fallback caso a playlist seja menor que o necessário ---
    if len(playlist_final) < num_recommendations_needed:
        faltantes = num_recommendations_needed - len(playlist_final)
        amostras_extra = data_with_genres.sample(n=min(faltantes, len(data_with_genres)), random_state=42)
        playlist_final = pd.concat([playlist_final, amostras_extra], ignore_index=True)
        print("INFO: Gerando músicas adicionais para completar a playlist.")

    # --- Garantir as colunas finais ---
    cols_necessarias = ['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features
    for c in cols_necessarias:
        if c not in playlist_final.columns:
            playlist_final[c] = None
    playlist_final = playlist_final[cols_necessarias]

    # --- Reação da persona ---
    if not playlist_final.empty:
        nomes_musicas = [f"{row['name']} - {row['artists']}" for _, row in playlist_final.iterrows()]
        reacao = chamada_api_retry(obter_reacao_persona, nomes_musicas, persona_gerada, novo_usuario)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    # --- Retorno final ---
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

# Dicionário para mapear gêneros descritivos para gêneros do dataset
generos_mapeamento = {
    'rock': ['rock', 'classic rock', 'hard rock', 'rock and roll', 'alternative rock'],
    'pop': ['pop', 'indie pop', 'pop rock', 'synth-pop', 'electropop'],
    'electronic': ['edm', 'dance pop', 'electronic', 'house', 'dubstep', 'techno', 'trance', 'electronica'],
    'hip hop': ['hip hop', 'rap', 'trap', 'drill'],
    'r&b': ['r&b', 'soul'],
    'jazz': ['jazz', 'smooth jazz', 'swing'],
    'folk': ['folk', 'singer-songwriter'],
    'metal': ['metal', 'heavy metal', 'death metal', 'black metal', 'thrash metal'],
    'classical': ['classical', 'symphony', 'orchestral'],
    'blues': ['blues', 'delta blues', 'electric blues'],
    'indie': ['indie pop', 'indie rock', 'indie folk'],
    'country': ['country', 'alt-country'],
    'dance': ['dance pop', 'edm', 'house'],
    'disco': ['disco', 'funk'],
    'reggae': ['reggae', 'dub'],
    'latin': ['latin', 'reggaeton', 'salsa', 'bachata'],
    'k-pop': ['k-pop'],
    'j-pop': ['j-pop'],
    'synthpop': ['synthpop', 'synth-pop', 'electropop', 'electro pop'],
    'alternative': ['alternative', 'alt rock', 'alt pop'],
}

label_cols = ['gender', 'country']

# DataFrame unificado de gêneros (do seu módulo de preprocessamento)
df_generos = criar_df_generos_unificado()

if df_generos is not None:
    novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa(
        interacoes_usuarios_artistas=pd.read_csv('./datasets/user_artists.csv', sep='\t'),
        dados_artistas=pd.read_csv('./datasets/artists.csv', sep='\t'),
        caracteristicas_por_artistadata_by_artist=pd.read_csv('./datasets/data_by_artist.csv'),
        musicas_base=pd.read_csv('./datasets/data.csv'),
        musicas_com_generos=df_generos,
        musical_features=musical_features,
        label_cols=label_cols,
        preference_map=preference_map,
        generos_mapeamento=generos_mapeamento,
        num_recommendations_needed=8,
        top_k_clusters=6,
    )

    print("Novo usuário:", novo_usuario)
    print("Persona gerada:", persona)
    print_playlist_bonita(playlist)
    if reacao:
        print("\n📝 Reação da persona:")
        print(reacao)
else:
    print("Erro: df_generos não foi criado. Verifique o preprocessamento.")