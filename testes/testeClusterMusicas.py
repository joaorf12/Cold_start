import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from openai import OpenAI
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

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

def criar_persona_chatgpt(usuario, preference_map):
    preferencias_numericas = {k: (preference_map[k][v] if v in preference_map[k] else v) for k,v in usuario.items() if k in preference_map}
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

# ==================== Função de recomendação completa ====================
# ==================== Função corrigida ====================
def gerar_recomendacao_completa_com_plot(client,
                                         interacoes_usuarios_artistas,
                                         dados_artistas,
                                         caracteristicas_por_artistadata_by_artist,
                                         musicas_base,
                                         musicas_com_generos,
                                         musical_features,
                                         label_cols,
                                         preference_map,
                                         num_recommendations_needed=10):
    # --- Preencher valores faltantes ---
    for feat in musical_features:
        musicas_com_generos[feat] = musicas_com_generos[feat].fillna(musicas_com_generos[feat].mean())
    X_musicas = musicas_com_generos[musical_features]
    scaler_musicas = StandardScaler()
    X_musicas_scaled = scaler_musicas.fit_transform(X_musicas)

    # --- Determinar número de clusters via Silhouette ---
    silhouette_scores = []
    cluster_range = range(2, 11)
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(X_musicas_scaled)
        silhouette_scores.append(silhouette_score(X_musicas_scaled, labels))
    n_clusters_final = cluster_range[np.argmax(silhouette_scores)]

    kmeans_musicas = KMeans(n_clusters=n_clusters_final, random_state=42, n_init='auto')
    musicas_com_generos['cluster'] = kmeans_musicas.fit_predict(X_musicas_scaled)

    # --- Novo usuário ---
    novo_usuario = gerar_novo_usuario_aleatorio(client)
    persona_gerada = criar_persona_chatgpt(novo_usuario, preference_map)

    user_vector = np.array([novo_usuario[feat] if isinstance(novo_usuario[feat], (int,float))
                            else preference_map[feat][novo_usuario[feat].lower()]
                            for feat in musical_features])
    user_vector_scaled = scaler_musicas.transform([user_vector])

    # --- Determinar clusters mais próximos dinamicamente ---
    distances = kmeans_musicas.transform(user_vector_scaled)
    num_nearest_clusters = max(1, int(len(kmeans_musicas.cluster_centers_) / 3))
    nearest_cluster_indices = distances.argsort()[0][:num_nearest_clusters]

    # --- Filtragem de músicas ---
    musicas_filtradas = musicas_com_generos[musicas_com_generos['cluster'].isin(nearest_cluster_indices)].copy()
    if musicas_filtradas.empty:
        musicas_filtradas = musicas_com_generos.copy()
    pop_threshold = musicas_filtradas['popularity'].quantile(0.8)
    musicas_filtradas = musicas_filtradas[musicas_filtradas['popularity'] >= pop_threshold]
    if musicas_filtradas.empty:
        musicas_filtradas = musicas_com_generos.copy()

    musicas_filtradas['distancia_perfil'] = np.linalg.norm(musicas_filtradas[musical_features].values - user_vector, axis=1)
    cols_necessarias = ['id','name','artists','popularity','artist_genres'] + musical_features
    musicas_recomendadas_final = musicas_filtradas.sort_values(by=['distancia_perfil','popularity'], ascending=[True,False]).head(num_recommendations_needed)[cols_necessarias]

    # --- Reação da persona ---
    if not musicas_recomendadas_final.empty:
        nomes_musicas = musicas_recomendadas_final['name'].tolist()
        reacao = obter_reacao_persona(nomes_musicas, persona_gerada)
    else:
        reacao = "🎧 Nenhuma música recomendada."

    # --- Plot dos clusters ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_musicas_scaled)
    plt.figure(figsize=(10,6))
    for cluster_id in range(n_clusters_final):
        idx = musicas_com_generos['cluster'] == cluster_id
        cluster_points = X_pca[idx]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], alpha=0.6, label=f"Cluster {cluster_id}")
    user_pca = pca.transform(user_vector_scaled)
    plt.scatter(user_pca[0,0], user_pca[0,1], c='red', s=200, marker='X', label='Usuário')
    plt.title("Clusterização das Músicas")
    plt.xlabel("Componente PCA 1")
    plt.ylabel("Componente PCA 2")
    plt.legend()
    plt.grid(True)
    plt.show()

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

novo_usuario, persona, playlist, reacao = gerar_recomendacao_completa_com_plot(
    client=client,
    interacoes_usuarios_artistas=pd.read_csv('../datasets/user_artists.csv', sep='\t'),
    dados_artistas=pd.read_csv('../datasets/artists.csv'),
    caracteristicas_por_artistadata_by_artist=pd.read_csv('../datasets/data_by_artist.csv'),
    musicas_base=pd.read_csv('../datasets/data.csv'),
    musicas_com_generos=pd.read_csv('../datasets/data_w_genres.csv'),
    musical_features=musical_features,
    label_cols=label_cols,
    preference_map=preference_map,
    num_recommendations_needed=10
)

print("Novo usuário:", novo_usuario)
print("Persona gerada:", persona)
print_playlist_bonita(playlist)
if reacao:
    print("\n📝 Reação da persona:")
    print(reacao)
