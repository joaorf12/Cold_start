import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
import os
from dotenv import load_dotenv
from collections import Counter

# ==================== Configurações ====================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==================== Funções auxiliares ====================
def print_playlist_bonita(playlist):
    if playlist.empty:
        print("🎧 Nenhuma música recomendada.")
        return

    cols = ['name', 'artists', 'popularity', 'artist_genres']
    data = playlist[cols].copy()
    data['artist_genres'] = data['artist_genres'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    name_width = max(data['name'].str.len().max(), len("Nome")) + 2
    artist_width = max(data['artists'].str.len().max(), len("Artista(s)")) + 2
    pop_width = max(len(str(data['popularity'].max())), len("Popularidade")) + 2
    genres_width = max(data['artist_genres'].str.len().max(), len("Gêneros")) + 2

    header = f"{'Nome'.ljust(name_width)}{'Artista(s)'.ljust(artist_width)}{'Popularidade'.ljust(pop_width)}{'Gêneros'.ljust(genres_width)}"
    print("\n🎧 Playlist Recomendada:\n")
    print(header)
    print("-" * len(header))

    for idx, row in data.iterrows():
        print(f"{row['name'].ljust(name_width)}{row['artists'].ljust(artist_width)}{str(row['popularity']).ljust(pop_width)}{row['artist_genres'].ljust(genres_width)}")


def extrair_dict_resposta(texto):
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("Não foi possível encontrar um dicionário na resposta.")
    return match.group(0)


def gerar_novo_usuario_aleatorio(client):
    prompt = """
Gere um novo usuário aleatório com características similares a este exemplo:

novo_usuario = {
'age': 16,
'gender': 'f',
'country': 'Brazil',
'danceability': 'baixo',
'energy': 'baixo',
'loudness': 'baixo',
'valence': 'baixo',
'tempo': 'baixo',
'acousticness': 'baixo',
'instrumentalness': 'baixo',
'liveness': 'baixo',
'speechiness': 'baixo'
}

Valores variados, seguindo padrão 'baixo', 'medio', 'alto' para as características musicais,
idade entre 13 e 50 anos, gênero 'm' ou 'f', country em ['Brazil','USA','Germany','UK','Japan'].
Retorne SOMENTE o dicionário em formato Python, sem explicações.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você gera perfis de usuários musicais."},
            {"role": "user", "content": prompt}
        ]
    )
    conteudo = response.choices[0].message.content.strip()
    try:
        dict_text = extrair_dict_resposta(conteudo)
        novo_usuario = ast.literal_eval(dict_text)
        if not isinstance(novo_usuario, dict):
            raise ValueError("Conteúdo não é um dicionário")
    except Exception as e:
        print("Erro ao converter a resposta em dict:", e)
        print("Conteúdo recebido:", conteudo)
        return None
    return novo_usuario


def mapear_preferencias(usuario, preference_map):
    preferencias_mapeadas = {}
    for chave, valor in usuario.items():
        if chave in preference_map:
            preferencias_mapeadas[chave] = preference_map[chave][valor]
    return preferencias_mapeadas


def criar_persona_chatgpt(usuario, preference_map):
    preferencias_numericas = mapear_preferencias(usuario, preference_map)
    prompt = f"""
Você é um especialista em perfis musicais.
A partir dos seguintes dados do usuário e suas preferências musicais mapeadas, crie uma persona detalhada:

- Idade: {usuario['age']}
- Gênero: {usuario['gender']}
- País: {usuario['country']}
- Preferências originais: {usuario}
- Preferências numéricas: {preferencias_numericas}

Descreva:
1. Nome fictício da persona
2. Estilo musical favorito
3. Artistas/bandas preferidas
4. Contextos em que costuma ouvir música
5. Personalidade e hábitos de consumo musical
"""
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um criador de personas musicais realistas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )
    return resposta.choices[0].message.content


def obter_reacao_persona(musicas, persona):
    if not musicas:
        return "Nenhuma música recomendada para avaliação."

    musicas_str = "\n".join([f"{i+1}. {m}" for i, m in enumerate(musicas)])
    prompt = f"""
Você deve interpretar a seguinte persona: {persona}.
Reaja de forma coerente às músicas recomendadas, comentando cada uma separadamente.
Depois, dê uma nota de 1 a 10 sobre cada música e uma média final.

Músicas:
{musicas_str}
"""
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você responde sempre no tom da persona."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )
    return resposta.choices[0].message.content.strip()

# ==================== Função principal ====================
def gerar_recomendacao_completa(client, interacoes_usuarios_artistas, dados_artistas,
                                caracteristicas_por_artistadata_by_artist, musicas_base,
                                musicas_com_generos, musical_features, label_cols,
                                preference_map, num_recommendations_needed=10):

    # --- Limpeza e padronização ---
    dfs_cols = [
        (dados_artistas,'name','name_clean'),
        (caracteristicas_por_artistadata_by_artist,'artists','artists_clean'),
        (musicas_base,'artists','artists_clean'),
        (musicas_com_generos,'artists','artists_clean')
    ]
    for df, col_in, col_out in dfs_cols:
        df[col_out] = df[col_in].str.lower().str.replace(r'[^\w\s]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()

    musicas_com_generos['genres'] = musicas_com_generos['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x,str) and x.startswith('[') else ([] if pd.isna(x) else [str(x)]))

    aggregated_genres = musicas_com_generos.groupby('artists_clean')['genres'].apply(
        lambda x: list(set(g for sublist in x for g in sublist))
    ).reset_index().rename(columns={'genres':'artist_genres'})

    data_with_genres = pd.merge(musicas_base, aggregated_genres, on='artists_clean', how='left')
    data_with_genres['artist_genres'] = data_with_genres['artist_genres'].fillna('').apply(lambda x: x if isinstance(x,list) else [])

    # --- Simulação de perfil ---
    np.random.seed(42)
    unique_users = interacoes_usuarios_artistas['userID'].unique()
    fake_users = pd.DataFrame({
        'userID': unique_users,
        'age': np.random.randint(15,60,len(unique_users)),
        'gender': np.random.choice(['m','f'],len(unique_users)),
        'country': np.random.choice(['Brazil','USA','Germany','UK','Japan'],len(unique_users))
    })
    merged = pd.merge(interacoes_usuarios_artistas, fake_users, on='userID')
    top_artists = merged.loc[merged.groupby('userID')['weight'].idxmax()]
    top_artists = pd.merge(top_artists, dados_artistas[['id','name','name_clean']], left_on='artistID', right_on='id', how='left')
    top_artists = pd.merge(top_artists, caracteristicas_por_artistadata_by_artist, left_on='name_clean', right_on='artists_clean', how='left')

    # --- Clusterização ---
    X_features = ['age','gender','country'] + [f for f in musical_features if f in top_artists.columns]
    for feature in X_features:
        if feature not in ['age','gender','country']:
            top_artists[feature] = top_artists[feature].fillna(top_artists[feature].mean())

    encoders = {col: LabelEncoder().fit(top_artists[col]) for col in label_cols}
    for col in label_cols:
        top_artists[col] = encoders[col].transform(top_artists[col])

    X = top_artists[X_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    cluster_range = range(2,11)
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    best_idx = np.argmax(silhouette_scores)
    n_clusters_final = cluster_range[best_idx]

    kmeans = KMeans(n_clusters=n_clusters_final, random_state=42, n_init='auto')
    top_artists['cluster'] = kmeans.fit_predict(X_scaled)

    # --- Usuário e persona ---
    novo_usuario = gerar_novo_usuario_aleatorio(client)
    persona_gerada = criar_persona_chatgpt(novo_usuario, preference_map)

    new_user_data = {}
    for key in ['age','gender','country']:
        if key in label_cols:
            new_user_data[key] = encoders[key].transform([novo_usuario[key]])[0]
        else:
            new_user_data[key] = novo_usuario[key]
    for feat in musical_features:
        if feat in novo_usuario:
            val = novo_usuario[feat]
            if isinstance(val,str):
                new_user_data[feat] = preference_map[feat][val.lower()]
            else:
                new_user_data[feat] = val
        else:
            new_user_data[feat] = top_artists[feat].mean()
    new_user_df = pd.DataFrame([new_user_data])[X_features]
    new_user_scaled = scaler.transform(new_user_df)

    # --- Clusters mais próximos dinamicamente ---
    distances = kmeans.transform(new_user_scaled)[0]
    min_distance = distances.min()
    threshold = min_distance * 1.2  # 20% acima da distância mínima
    nearest_cluster_indices = np.where(distances <= threshold)[0]

    print(f"\nO novo usuário está relacionado aos clusters: {nearest_cluster_indices}")
    print(f"O usuário foi aderido a {len(nearest_cluster_indices)} clusters.")

    # --- Filtragem de músicas ---
    artistas_clusters = top_artists[top_artists['cluster'].isin(nearest_cluster_indices)]['name_clean'].unique()
    musicas_filtradas = data_with_genres[data_with_genres['artists_clean'].isin(artistas_clusters)].copy()

    if musicas_filtradas.empty:
        musicas_filtradas = data_with_genres.copy()

    pop_threshold = musicas_filtradas['popularity'].quantile(0.8)
    musicas_filtradas = musicas_filtradas[musicas_filtradas['popularity']>=pop_threshold]

    if musicas_filtradas.empty:
        musicas_filtradas = data_with_genres.copy()

    persona_generos = list(set(re.findall(
        r'\b(rock|pop|hip hop|rap|r&b|country|soul|indie|electropop|synthpop)\b', persona_gerada.lower()
    )))
    if persona_generos:
        musicas_filtradas_por_genero = musicas_filtradas[musicas_filtradas['artist_genres'].apply(
            lambda gs: any(g in gs for g in persona_generos)
        )]
        if not musicas_filtradas_por_genero.empty:
            musicas_filtradas = musicas_filtradas_por_genero

    user_vector = np.array([new_user_data[feat] for feat in musical_features])
    musicas_filtradas['distancia_perfil'] = np.linalg.norm(
        musicas_filtradas[musical_features].values - user_vector, axis=1
    )

    cols_necessarias = ['id','name','artists','popularity','artist_genres'] + musical_features
    musicas_filtradas = musicas_filtradas.sort_values(by=['distancia_perfil','popularity'], ascending=[True,False])
    musicas_recomendadas_final = musicas_filtradas.head(num_recommendations_needed)[cols_necessarias]

    if not musicas_recomendadas_final.empty:
        nomes_musicas = musicas_recomendadas_final['name'].tolist()
        reacao = obter_reacao_persona(nomes_musicas, persona_gerada)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    return novo_usuario, persona_gerada, musicas_recomendadas_final, reacao

# ==================== Uso da função ====================
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

novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa(
    client,
    pd.read_csv('./datasets/user_artists.csv', sep='\t'),
    pd.read_csv('./datasets/artists.csv', sep='\t'),
    pd.read_csv('./datasets/data_by_artist.csv'),
    pd.read_csv('./datasets/data.csv'),
    pd.read_csv('./datasets/data_w_genres.csv'),
    musical_features,
    label_cols,
    preference_map,
    num_recommendations_needed=10
)

print("Novo usuário:", novo_usuario)
print("Persona gerada:", persona)
print_playlist_bonita(playlist)
if reacao:
    print("\n📝 Reação da persona:")
    print(reacao)
