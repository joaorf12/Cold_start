import pandas as pd
import numpy as np
import ast  # Import para literal_eval
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()
# Inicializa cliente (defina sua chave de API no ambiente)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(client)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def extrair_dict_resposta(texto):
    # Extrai o texto que começa com { e termina com }
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("Não foi possível encontrar um dicionário na resposta.")
    dict_text = match.group(0)
    return dict_text

# Funação para gerar um novo usuário
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

    Por favor, gere valores variados, mas com as mesmas chaves e valores plausíveis para cada uma, seguindo o padrão: 'baixo', 'medio', 'alto' para as características musicais, e idade entre 13 e 50 anos, gênero 'm' ou 'f'.

    **IMPORTANTE:** O campo 'country' deve ser apenas um dos seguintes países: 'Brazil', 'USA', 'Germany', 'UK', 'Japan'.

    Retorne SOMENTE o dicionário no formato Python, sem explicações.
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

# Função para mapear preferências para valores numéricos
def mapear_preferencias(usuario, preference_map):
    preferencias_mapeadas = {}
    for chave, valor in usuario.items():
        if chave in preference_map:
            preferencias_mapeadas[chave] = preference_map[chave][valor]
    return preferencias_mapeadas

# Função para criar a persona via ChatGPT
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
    # Se for lista, transforma em string concatenada
    if isinstance(musicas, list):
        musicas_str = ', '.join(musicas)
    else:
        musicas_str = musicas  # já string pronta

    prompt = f"""
    Você deve interpretar a seguinte persona: {persona}.
    Reaja de forma coerente às músicas recomendadas:
    {musicas_str}

    Depois, dê uma nota de 1 a 10 sobre as recomendações.
    """

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você responde sempre no tom da persona."},
            {"role": "user", "content": prompt}
        ]
    )

    return resposta.choices[0].message.content.strip()

# ========== 1. Carga de Dados ==========
# Carrega os datasets
user_artists = pd.read_csv('./datasets/user_artists.csv', sep='\t')
artists = pd.read_csv('./datasets/artists.csv', sep='\t')
data_by_artist = pd.read_csv('./datasets/data_by_artist.csv')
data = pd.read_csv('./datasets/data.csv')
data_w_genres = pd.read_csv('./datasets/data_w_genres.csv')  # Carrega dados com gêneros

# Simplified artist name cleaning (without unidecode/re as they are not available)
# Garante que as colunas 'artists' existam antes de tentar limpá-las.
artists['name_clean'] = artists['name'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ',
                                                                                                        regex=True).str.strip()
data_by_artist['artists_clean'] = data_by_artist['artists'].str.lower().str.replace(r'[^\w\s]', '',
                                                                                    regex=True).str.replace(r'\s+', ' ',
                                                                                                            regex=True).str.strip()
data['artists_clean'] = data['artists'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ',
                                                                                                        regex=True).str.strip()
# ADICIONADO/CORRIGIDO: Limpeza para data_w_genres
data_w_genres['artists_clean'] = data_w_genres['artists'].str.lower().str.replace(r'[^\w\s]', '',
                                                                                  regex=True).str.replace(r'\s+', ' ',
                                                                                                          regex=True).str.strip()

# ========== 2. Simulação de Perfil de Usuário ==========
np.random.seed(42)
unique_users = user_artists['userID'].unique()
fake_users = pd.DataFrame({
    'userID': unique_users,
    'age': np.random.randint(15, 60, size=len(unique_users)),
    'gender': np.random.choice(['m', 'f'], size=len(unique_users)),
    'country': np.random.choice(['Brazil', 'USA', 'Germany', 'UK', 'Japan'], size=len(unique_users)),
})

# ========== 3. Combinação e Pré-processamento ==========
merged = pd.merge(user_artists, fake_users, on='userID')
top_artists = merged.loc[merged.groupby('userID')['weight'].idxmax()]

top_artists = pd.merge(top_artists, artists[['id', 'name', 'name_clean']], left_on='artistID', right_on='id',
                       how='left')

top_artists = pd.merge(top_artists, data_by_artist, left_on='name_clean', right_on='artists_clean', how='left')

# NOVO/CORRIGIDO: Mescla dados de músicas com gêneros
# A linha problematica data_w_genres['song_id'] = ... FOI REMOVIDA.
# Tratamento para garantir que 'genres' em data_w_genres seja uma lista antes do merge
# Ajustado para lidar com valores não-string ou nulos de forma mais robusta.
data_w_genres['genres'] = data_w_genres['genres'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) and x.startswith('[') else (
        [] if pd.isna(x) else [str(x)]))

# Agrupa e junta os gêneros para cada artista para ter uma lista única de gêneros por artista.
aggregated_genres = data_w_genres.groupby('artists_clean')['genres'].apply(
    lambda x: list(set(g for sublist in x for g in sublist))).reset_index()
aggregated_genres.rename(columns={'genres': 'artist_genres'}, inplace=True)

# Mescla `data` com os gêneros agregados dos artistas usando 'artists_clean'
data_with_genres = pd.merge(data, aggregated_genres, on='artists_clean', how='left')
data_with_genres['artist_genres'] = data_with_genres['artist_genres'].fillna('').apply(
    lambda x: x if isinstance(x, list) else [])

# ========== 4. Clustering usando dados demográficos E musicais ==========

musical_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

for feature in musical_features:
    if feature not in top_artists.columns:
        print(f"Aviso: Característica musical '{feature}' não encontrada. Verifique data_by_artist.csv.")
        musical_features.remove(feature)

for feature in musical_features:
    if top_artists[feature].isnull().any():
        top_artists[feature] = top_artists[feature].fillna(top_artists[feature].mean())

label_cols = ['gender', 'country']
encoders = {col: LabelEncoder().fit(top_artists[col]) for col in label_cols}
for col in label_cols:
    top_artists[col] = encoders[col].transform(top_artists[col])

X_features = ['age', 'gender', 'country'] + musical_features
X = top_artists[X_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Explorar diferentes números de clusters (n_clusters)
print("\n--- Explorando o número de Clusters (n_clusters) ---")
silhouette_scores = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans_test = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans_test.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    print(f"n_clusters = {n_clusters}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Score para diferentes números de clusters')
plt.xlabel('Número de Clusters (n_clusters)')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.grid(True)
#plt.show()

n_clusters_final = 10
print(f"\nNúmero de clusters escolhido para o modelo final: {n_clusters_final} (para maior granularidade).")

kmeans = KMeans(n_clusters=n_clusters_final, random_state=42, n_init='auto')
top_artists['cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, top_artists['cluster'])
print(f"✅ Silhouette Score para o n_clusters escolhido ({n_clusters_final}): {sil_score:.4f}")

print("\n--- Características Musicais Médias por Cluster ---")
X_descaled = scaler.inverse_transform(X_scaled)
df_descaled = pd.DataFrame(X_descaled, columns=X_features)
df_descaled['cluster'] = top_artists['cluster']

cluster_musical_profiles = df_descaled.groupby('cluster')[musical_features].mean()

cluster_musical_profiles_melted = cluster_musical_profiles.reset_index().melt(
    id_vars='cluster', var_name='Característica Musical', value_name='Valor Médio'
)

plt.figure(figsize=(15, 8))
sns.barplot(
    data=cluster_musical_profiles_melted,
    x='Característica Musical',
    y='Valor Médio',
    hue='cluster',
    palette='viridis'
)
plt.title('Perfil Musical Médio dos Clusters')
plt.xlabel('Característica Musical')
plt.ylabel('Valor Médio (Desescalado)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster')
plt.tight_layout()
#plt.show()

# ========== 5. Recomendação para novo usuário (Cold-Start) ==========

# BLOCO 1: Ajuste dos Valores no `preference_map` e Aumento dos Clusters Próximos
preference_map = {
    'danceability': {'alto': 0.85, 'medio': 0.5, 'baixo': 0.15},
    'energy': {'alto': 0.85, 'medio': 0.5, 'baixo': 0.15},
    'loudness': {'alto': -4.0, 'medio': -15.0, 'baixo': -35.0},
    'valence': {'alto': 0.8, 'medio': 0.5, 'baixo': 0.2},
    'acousticness': {'alto': 0.1, 'medio': 0.4, 'baixo': 0.7},
    'instrumentalness': {'alto': 0.8, 'medio': 0.3, 'baixo': 0.02},
    'liveness': {'alto': 0.75, 'medio': 0.3, 'baixo': 0.05},
    'speechiness': {'alto': 0.6, 'medio': 0.2, 'baixo': 0.03},
    'tempo': {'alto': 160.0, 'medio': 110.0, 'baixo': 65.0}
}

# novo_usuario = {
#     'age': 23,
#     'gender': 'm',
#     'country': 'Brazil',
#     'danceability': 'alto',
#     'energy': 'baixo',
#     'loudness': 'alto',
#     'valence': 'alto',
#     'tempo': 'alto',
#     'acousticness': 'baixo',
#     'instrumentalness': 'baixo',
#     'liveness': 'baixo',
#     'speechiness': 'baixo',
# }

# novo_usuario = {
#     'age': 16,
#     'gender': 'f',
#     'country': 'Brazil',
#     'danceability': 'baixo',
#     'energy': 'baixo',
#     'loudness': 'baixo',
#     'valence': 'baixo',
#     'tempo': 'baixo',
#     'acousticness': 'baixo', # Mapeia para 0.7 (alta acousticness)
#     'instrumentalness': 'baixo', # Mapeia para 0.02 (baixa instrumentalness = com vocais)
#     'liveness': 'baixo',
#     'speechiness': 'baixo', # Mapeia para 0.03 (baixa speechiness = mais melódica/cantada)
# }

novo_usuario = gerar_novo_usuario_aleatorio(client)
print(novo_usuario)

# Exemplo de uso
persona_gerada = criar_persona_chatgpt(novo_usuario, preference_map)
print(persona_gerada)

new_user_data = {}

for key in ['age', 'gender', 'country']:
    if key in novo_usuario:
        if key in label_cols:
            new_user_data[key] = encoders[key].transform([novo_usuario[key]])[0]
        else:
            new_user_data[key] = novo_usuario[key]

for feature in musical_features:
    if feature in novo_usuario:
        user_preference = novo_usuario[feature]
        if isinstance(user_preference, str) and user_preference.lower() in preference_map[feature]:
            new_user_data[feature] = preference_map[feature][user_preference.lower()]
        elif isinstance(user_preference, (int, float)):
            new_user_data[feature] = user_preference
        else:
            print(f"Aviso: Preferência '{user_preference}' para '{feature}' não reconhecida. Usando média.")
            new_user_data[feature] = top_artists[feature].mean()
    else:
        new_user_data[feature] = top_artists[feature].mean()

new_user_df = pd.DataFrame([new_user_data])
new_user_df = new_user_df[X_features]
new_user_scaled = scaler.transform(new_user_df)

num_nearest_clusters = 2  # O usuário estará relacionado a TODOS os clusters finais

distances = euclidean_distances(new_user_scaled, kmeans.cluster_centers_)
nearest_cluster_indices = distances.argsort()[0][:num_nearest_clusters]

print(f"\nO novo usuário está relacionado aos clusters: {nearest_cluster_indices}")
print(f"O usuário foi aderido a {len(nearest_cluster_indices)} clusters.")

all_artists_from_nearest_clusters = pd.Series(dtype='object')

for cluster_idx in nearest_cluster_indices:
    artists_in_cluster = top_artists[top_artists['cluster'] == cluster_idx]['name_clean']
    all_artists_from_nearest_clusters = pd.concat([all_artists_from_nearest_clusters, artists_in_cluster])

all_artists_from_nearest_clusters_unique = all_artists_from_nearest_clusters.unique()

# Filtrar músicas dos clusters relevantes, agora com informações de gênero
musicas_filtradas = data_with_genres[data_with_genres['artists_clean'].isin(all_artists_from_nearest_clusters_unique)]

# Remover músicas sem gênero ou com lista de gênero vazia, se houver
musicas_filtradas = musicas_filtradas[musicas_filtradas['artist_genres'].apply(lambda x: len(x) > 0)]

# BLOCO NOVO: Injeção de Gêneros e Diversificação Aprimorada
musicas_recomendadas_final = pd.DataFrame()
recommended_artists_set = set()
recommended_genres_set = set()  # NOVO: Para rastrear gêneros já incluídos
num_recommendations_needed = 10

# 1. Obter os gêneros mais relevantes (frequentes) dentro das músicas filtradas
all_genres_in_filtered_songs = []
for genres_list in musicas_filtradas['artist_genres']:  # Usar 'artist_genres'
    all_genres_in_filtered_songs.extend(genres_list)

from collections import Counter

genre_counts = Counter(all_genres_in_filtered_songs)
most_common_genres = [genre for genre, count in genre_counts.most_common() if
                      genre != '']  # Excluir gêneros vazios/desconhecidos

# 2. Tentar pegar as músicas mais populares por gênero primeiro
for genre in most_common_genres:
    if len(musicas_recomendadas_final) >= num_recommendations_needed:
        break

    # Se este gênero já foi incluído para diversificação, pular
    if genre in recommended_genres_set:
        continue

    # Filtrar músicas deste gênero, ordenar por popularidade e pegar a primeira não recomendada
    genre_specific_songs = musicas_filtradas[
        musicas_filtradas['artist_genres'].apply(lambda x: genre in x)].sort_values(by='popularity', ascending=False)

    for idx, row in genre_specific_songs.iterrows():
        # Lidar com a coluna 'artists' que pode ser uma string de lista
        artist_name_list_str = row['artists']
        # Tenta converter a string de lista para uma lista real
        try:
            artist_name_list = ast.literal_eval(artist_name_list_str)
        except (ValueError, SyntaxError):
            # Fallback se não for uma string de lista válida
            artist_name_list = [artist_name_list_str] if isinstance(artist_name_list_str, str) else []

        main_artist = artist_name_list[0] if artist_name_list else "Unknown Artist"

        # Garante que a música não foi recomendada E que o artista principal não foi recomendado
        if (
        row['id'] not in musicas_recomendadas_final['id'].values if not musicas_recomendadas_final.empty else True) and \
                (main_artist not in recommended_artists_set):
            musicas_recomendadas_final = pd.concat([musicas_recomendadas_final, pd.DataFrame([row])], ignore_index=True)
            recommended_artists_set.add(main_artist)
            recommended_genres_set.add(genre)  # Marca o gênero como "usado" para diversificação
            break  # Pegar apenas uma música por este gênero principal neste ciclo

# 3. Preencher o restante com as músicas mais populares, garantindo diversidade de artista
# (similar à lógica anterior, mas agindo como um fallback)
if len(musicas_recomendadas_final) < num_recommendations_needed:
    remaining_needed = num_recommendations_needed - len(musicas_recomendadas_final)

    # Excluir músicas já recomendadas e artistas já recomendados
    remaining_songs_pool = musicas_filtradas[
        ~musicas_filtradas['id'].isin(
            musicas_recomendadas_final['id'].tolist() if not musicas_recomendadas_final.empty else [])
    ].sort_values(by='popularity', ascending=False)

    for idx, row in remaining_songs_pool.iterrows():
        if len(musicas_recomendadas_final) >= num_recommendations_needed:
            break

        # Lidar com a coluna 'artists' novamente
        artist_name_list_str = row['artists']
        try:
            artist_name_list = ast.literal_eval(artist_name_list_str)
        except (ValueError, SyntaxError):
            artist_name_list = [artist_name_list_str] if isinstance(artist_name_list_str, str) else []
        main_artist = artist_name_list[0] if artist_name_list else "Unknown Artist"

        if main_artist not in recommended_artists_set:
            musicas_recomendadas_final = pd.concat([musicas_recomendadas_final, pd.DataFrame([row])], ignore_index=True)
            recommended_artists_set.add(main_artist)

musicas_formatadas = ""

# 🖨️ Exibe as recomendações finais
if not musicas_recomendadas_final.empty:
    print(
        f"\n🎧 Músicas recomendadas para novo usuário (com diversidade de artistas E gênero - {len(recommended_genres_set)} gêneros únicos):")
    for idx, row in musicas_recomendadas_final.iterrows():
        genres_display = ', '.join(row['artist_genres']) if row['artist_genres'] else 'N/A'  # Usar 'artist_genres'
        linha = f"- {row['name']} (Artista: {row['artists']}, Popularidade: {row['popularity']}, Gêneros: {genres_display})"
        print(linha)
        musicas_formatadas += linha + "\n"
else:
    print("⚠️ Nenhuma música encontrada para os artistas dos clusters ou gêneros relacionados.")

# Chatgpt Avalia

# Avaliação com várias personas
# Personas para simulação
# personas = [
#     "Um adolescente fã de hits virais do TikTok",
#     "Um crítico musical que só gosta de jazz e blues",
#     "Um fã hardcore de heavy metal",
#     "Um DJ de música eletrônica dos anos 90",
#     "Uma senhora nostálgica por músicas dos anos 60"
# ]
#
# # Executa para cada persona
# for p in personas:
#     print(f"\n--- Persona: {p} ---")
#     print(obter_reacao_persona(musicas_recomendadas_final, p))

# Escolha da persona
#persona = "Um fã de rock clássico que não suporta música pop moderna."

# Chama a função e mostra no console
opiniao = obter_reacao_persona(musicas_formatadas, persona_gerada)
print(opiniao)