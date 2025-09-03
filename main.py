import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from preprocessamento import criar_df_generos_unificado
from visualizacao import plotar_clusters, print_playlist_bonita
from chamadasGemini import (
    chamada_api_retry,
    gerar_novo_usuario_aleatorio,
    criar_persona_gemini,
    obter_reacao_persona,
)


def filtrar_playlist(df_musicas, persona, top_n=20):
    """
    Filtra a playlist de acordo com a persona:
    - Prioriza gêneros que batem com a persona
    - Ajusta por energy e tempo
    - Mantém fallback caso não haja músicas suficientes nos clusters
    """

    # ======================
    # 1. Mapear gêneros da persona
    # ======================
    generos_persona = [g.lower() for g in persona['genres']]  # ex: ['indie electronica', 'indie rock', ...]

    # Padronizar gêneros do dataset
    df_musicas['genres_lower'] = df_musicas['artist_genres'].apply(lambda gs: [g.lower() for g in gs])

    # ======================
    # 2. Score de correspondência de gênero
    # ======================
    def genero_score(musica_genres):
        # número de gêneros da música que batem com a persona
        return len(set(musica_genres) & set(generos_persona))

    df_musicas['genero_score'] = df_musicas['genres_lower'].apply(genero_score)

    # ======================
    # 3. Score de energia/tempo
    # ======================
    energia_map = {'baixo': 0.2, 'medio': 0.5, 'alto': 0.8}
    tempo_map = {'baixo': 60, 'medio': 100, 'alto': 140}

    energy_persona = energia_map[persona['energy']]
    tempo_persona = tempo_map[persona['tempo']]

    # Normalizar entre 0 e 1
    df_musicas['energy_score'] = 1 - abs(df_musicas['energy'] - energy_persona)
    df_musicas['tempo_score'] = 1 - abs(df_musicas['tempo'] - tempo_persona) / max(tempo_map.values())

    # ======================
    # 4. Score final
    # ======================
    df_musicas['final_score'] = (
            df_musicas['genero_score'] * 0.5 +
            df_musicas['energy_score'] * 0.25 +
            df_musicas['tempo_score'] * 0.25
    )

    # ======================
    # 5. Ordenar e pegar top N
    # ======================
    df_musicas_sorted = df_musicas.sort_values(by='final_score', ascending=False)

    return df_musicas_sorted.head(top_n)

# ==================== Função para gerar playlist personalizada ====================
def gerar_playlist_personalizada(top_artists, clusters_relacionados, novo_usuario, X_features_playlist, qtd_por_cluster=2):
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
    num_recommendations_needed=10,
    num_clusters_desejado=10,       # <-- você controla quantos clusters quer forçar
    top_k_clusters=6,               # <-- quantos clusters mais próximos do usuário usar
):
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

    def obter_generos_mapeados(generos_persona):
        generos_traduzidos = set()
        for genero_persona in generos_persona:
            genero_lower = genero_persona.lower()
            for chave, valores in generos_mapeamento.items():
                if chave in genero_lower:
                    generos_traduzidos.update(valores)
        return list(generos_traduzidos)

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

    # --- Simulação de perfil demográfico para usuários (para features categóricas) ---
    np.random.seed(42)
    unique_users = interacoes_usuarios_artistas['userID'].unique()
    fake_users = pd.DataFrame({
        'userID': unique_users,
        'age': np.random.randint(15, 60, len(unique_users)),
        'gender': np.random.choice(['m', 'f'], len(unique_users)),
        'country': np.random.choice(['Brazil', 'USA', 'Germany', 'UK', 'Japan'], len(unique_users)),
    })
    merged = pd.merge(interacoes_usuarios_artistas, fake_users, on='userID')

    # Para cada user, pega o artista mais forte (maior weight)
    top_artists = merged.loc[merged.groupby('userID')['weight'].idxmax()].copy()
    top_artists = pd.merge(
        top_artists,
        dados_artistas[['id', 'name', 'name_clean']],
        left_on='artistID',
        right_on='id',
        how='left'
    )
    top_artists = pd.merge(
        top_artists,
        caracteristicas_por_artistadata_by_artist,
        left_on='name_clean',
        right_on='artists_clean',
        how='left'
    )

    # Seleção das features:
    # - Para o MODELO (clustering): demográficas + musicais
    # - Para a PLAYLIST: apenas as musicais (compatível com mapas qualitativos)
    X_features_modelo = ['age', 'gender', 'country'] + [f for f in musical_features if f in top_artists.columns]
    X_features_playlist = [f for f in musical_features if f in top_artists.columns]

    # Preencher valores faltantes nas features do modelo
    for feature in X_features_modelo:
        if feature not in ['age', 'gender', 'country']:
            top_artists[feature] = top_artists[feature].fillna(top_artists[feature].mean())

    # Label Encoding para categóricas (gender, country)
    encoders = {}
    for col in label_cols:
        enc = LabelEncoder()
        top_artists[col] = enc.fit_transform(top_artists[col].astype(str))
        encoders[col] = enc

    # Normalização das features do modelo
    scaler_modelo = StandardScaler()
    X_scaled_modelo = scaler_modelo.fit_transform(top_artists[X_features_modelo])

    # =========================
    # HDBSCAN + KMeans fallback (para forçar mais clusters)
    # =========================
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='leaf'
    )
    top_artists['cluster_hdbscan'] = clusterer.fit_predict(X_scaled_modelo)

    # Separa dados válidos e ruído
    dados_sem_ruido = top_artists[top_artists['cluster_hdbscan'] != -1].copy()
    dados_ruido     = top_artists[top_artists['cluster_hdbscan'] == -1].copy()

    # Se não houver dados sem ruído, usa todo mundo no KMeans
    if dados_sem_ruido.empty:
        dados_sem_ruido = top_artists.copy()
        dados_ruido = pd.DataFrame(columns=top_artists.columns)

    # Força K clusters nos dados sem ruído
    kmeans = KMeans(n_clusters=num_clusters_desejado, random_state=42, n_init=10)
    dados_sem_ruido['cluster'] = kmeans.fit_predict(dados_sem_ruido[X_features_modelo])

    # Atribui o ruído ao cluster mais próximo
    if not dados_ruido.empty:
        dados_ruido['cluster'] = kmeans.predict(dados_ruido[X_features_modelo])

    # Junta tudo de volta
    top_artists = pd.concat([dados_sem_ruido, dados_ruido], ignore_index=True)
    print("Clusters finais (com fallback):")
    print(top_artists['cluster'].value_counts())

    # --- Usuário e persona (única vez, coerente com preference_map) ---
    novo_usuario = chamada_api_retry(gerar_novo_usuario_aleatorio)
    persona_info = chamada_api_retry(criar_persona_gemini, novo_usuario, preference_map)
    persona_gerada = persona_info['persona_text']
    persona_generos = persona_info['persona_genres']

    # Mapear gêneros textuais da persona para gêneros do dataset
    generos_para_filtro = obter_generos_mapeados(persona_generos)

    def padronizar_genero(g):
        return g.lower().replace("-", " ").strip()

    persona_generos = [padronizar_genero(g) for g in persona_generos]
    generos_para_filtro = [padronizar_genero(g) for g in generos_para_filtro]

    print("Generos da persona:", persona_generos)
    print("Generos mapeados:", generos_para_filtro)

    # Vetor numérico do novo usuário para as features musicais (para distância na playlist)
    # e também criar um vetor para distância até centróides de clusters (usa X_features_modelo):
    new_user_data_modelo = {}
    # demográficas
    for key in ['age', 'gender', 'country']:
        if key in label_cols:
            new_user_data_modelo[key] = encoders[key].transform([novo_usuario.get(key, 'm' if key=='gender' else 'USA')])[0] \
                if novo_usuario.get(key) in encoders[key].classes_ else 0
        else:
            # se não vier do gerador, usa média
            new_user_data_modelo[key] = top_artists[key].mean()

    # musicais (usa preference_map)
    for feat in musical_features:
        val = novo_usuario.get(feat, None)
        if isinstance(val, str):
            new_user_data_modelo[feat] = preference_map[feat][val.lower()]
        elif isinstance(val, (int, float)):
            new_user_data_modelo[feat] = val
        else:
            new_user_data_modelo[feat] = top_artists[feat].mean()

    new_user_df_modelo = pd.DataFrame([new_user_data_modelo])[X_features_modelo]
    new_user_scaled = scaler_modelo.transform(new_user_df_modelo)

    # Distância do novo usuário aos centróides de KMeans
    clusters_centroids = top_artists.groupby('cluster')[X_features_modelo].mean().values
    dists = np.linalg.norm(clusters_centroids - new_user_scaled, axis=1).flatten()

    # Seleciona os K clusters mais próximos
    all_cluster_labels = sorted(top_artists['cluster'].unique())
    ordem = np.argsort(dists)
    clusters_relacionados = [all_cluster_labels[i] for i in ordem[:min(top_k_clusters, len(all_cluster_labels))]]
    print(f"O novo usuário está relacionado aos clusters (top-{len(clusters_relacionados)}): {clusters_relacionados}")

    # === Playlist personalizada por proximidade (em cima de top_artists) ===
    playlist_seed = gerar_playlist_personalizada(
        top_artists=top_artists,
        clusters_relacionados=clusters_relacionados,
        novo_usuario=novo_usuario,
        X_features_playlist=X_features_playlist,
        qtd_por_cluster=max(1, num_recommendations_needed // max(1, len(clusters_relacionados)))
    )

    # Usa artistas da seed para filtrar faixas reais
    artistas_clusters = playlist_seed['name_clean'].unique() if not playlist_seed.empty else \
                        top_artists[top_artists['cluster'].isin(clusters_relacionados)]['name_clean'].unique()

    musicas_filtradas = data_with_genres[data_with_genres['artists_clean'].isin(artistas_clusters)].copy()
    if musicas_filtradas.empty:
        musicas_filtradas = data_with_genres.copy()

    # --- Filtro de popularidade ---
    if 'popularity' in musicas_filtradas.columns and not musicas_filtradas['popularity'].isna().all():
        pop_threshold = musicas_filtradas['popularity'].quantile(0.5)
        mf = musicas_filtradas[musicas_filtradas['popularity'] >= pop_threshold]
        if not mf.empty:
            musicas_filtradas = mf

    # --- Filtro por gêneros preferidos (fallback) ---
    if generos_para_filtro:
        musicas_por_genero = musicas_filtradas[musicas_filtradas['artist_genres'].apply(
            lambda gs: any(g in gs for g in generos_para_filtro)
        )]
        if len(musicas_por_genero) >= num_recommendations_needed // 2:
            musicas_filtradas = musicas_por_genero
        else:
            print("INFO: Pouca aderência por gênero nos clusters; buscando no dataset completo…")
            fallback_musicas = data_with_genres[data_with_genres['artist_genres'].apply(
                lambda gs: any(g in gs for g in generos_para_filtro)
            )].copy()
            if not fallback_musicas.empty and 'popularity' in fallback_musicas.columns:
                pop_thr_fb = fallback_musicas['popularity'].quantile(0.5)
                fb2 = fallback_musicas[fallback_musicas['popularity'] >= pop_thr_fb]
                if not fb2.empty:
                    fallback_musicas = fb2
            if not fallback_musicas.empty:
                musicas_filtradas = fallback_musicas

    # --- Ordenar por proximidade ao perfil musical e pontuar gênero ---
    user_vector_musical = np.array([new_user_data_modelo[feat] for feat in musical_features])
    musicas_filtradas = musicas_filtradas.copy()
    # garante colunas
    for feat in musical_features:
        if feat not in musicas_filtradas.columns:
            musicas_filtradas[feat] = top_artists[feat].mean()

    musicas_filtradas['distancia_perfil'] = np.linalg.norm(
        musicas_filtradas[musical_features].values - user_vector_musical, axis=1
    )
    musicas_filtradas['pontuacao_genero'] = 0.0
    if generos_para_filtro:
        musicas_filtradas['pontuacao_genero'] = musicas_filtradas['artist_genres'].apply(
            lambda gs: 1.0 if any(g in gs for g in generos_para_filtro) else -0.5
        )

    musicas_filtradas['distancia_ponderada'] = musicas_filtradas['distancia_perfil'] - musicas_filtradas['pontuacao_genero']

    # Colunas finais
    cols_necessarias = ['id', 'name', 'artists', 'popularity', 'artist_genres'] + musical_features
    musicas_filtradas = musicas_filtradas.sort_values(
        by=['distancia_ponderada', 'popularity'] if 'popularity' in musicas_filtradas.columns else ['distancia_ponderada'],
        ascending=[True, False] if 'popularity' in musicas_filtradas.columns else [True]
    )

    # Remove duplicadas por nome (ou por id, se preferir)
    if 'name' in musicas_filtradas.columns:
        musicas_filtradas = musicas_filtradas.drop_duplicates(subset=['name'], keep='first')

    persona_para_filtro = {
        'genres': persona_generos,
        'energy': novo_usuario['energy'],  # também precisa de energy
        'tempo': novo_usuario['tempo']  # e tempo
    }

    playlist_filtrada = filtrar_playlist(musicas_filtradas, persona_para_filtro, top_n=15)
    print(playlist_filtrada[['name', 'artists', 'genero_score', 'energy_score', 'tempo_score', 'final_score']])

    # --- Depois de calcular playlist_filtrada ---
    playlist_final = playlist_filtrada.copy()

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
        amostras_extra = data_with_genres.sample(faltantes, random_state=42)
        playlist_final = pd.concat([playlist_final, amostras_extra], ignore_index=True)

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
        num_recommendations_needed=8,
        num_clusters_desejado=10,     # << ajuste aqui para "usar mais clusters"
        top_k_clusters=6,             # << quantos clusters mais próximos considerar
    )

    print("Novo usuário:", novo_usuario)
    print("Persona gerada:", persona)
    print_playlist_bonita(playlist)
    if reacao:
        print("\n📝 Reação da persona:")
        print(reacao)
else:
    print("Erro: df_generos não foi criado. Verifique o preprocessamento.")
