import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

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

def plotar_clusters(top_artists, musical_features):
    # Calcula a média das features por cluster
    cluster_means = top_artists.groupby("cluster")[musical_features].mean()

    # --- Heatmap ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, cmap="viridis", cbar=True, fmt=".2f")
    plt.title("Médias das características musicais por cluster")
    plt.xlabel("Características musicais")
    plt.ylabel("Clusters")
    plt.show()

    # --- Barplot (médias por cluster) ---
    cluster_means.T.plot(kind="bar", figsize=(14, 6))
    plt.title("Média das características por cluster")
    plt.ylabel("Média normalizada")
    plt.xlabel("Características musicais")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # --- Boxplot (distribuição das features por cluster) ---
    df_melted = top_artists.melt(id_vars="cluster", value_vars=musical_features,
                                 var_name="Feature", value_name="Valor")
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_melted, x="Feature", y="Valor", hue="cluster")
    plt.title("Distribuição das características por cluster")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cluster")
    plt.tight_layout()
    plt.show()

    # --- PCA Scatter (redução para 2D) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(top_artists[musical_features])
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["cluster"] = top_artists["cluster"].values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="cluster", palette="tab10")
    plt.title("Clusters em 2D via PCA")
    plt.tight_layout()
    plt.show()

# Adicione esta função ao seu main.py, ou preferencialmente, no seu módulo 'visualizacao'
def plot_caracteristicas_dominantes(user_profiles_encoded, musical_features, scaler_modelo, k_ideal, top_n_features=5):
    """
    Calcula os centróides dos clusters, inverte o escalonamento para a média original,
    e plota as características musicais mais dominantes de cada cluster.
    """
    print("\n--- Analisando Centróides dos Clusters ---")

    # 1. Calcular os centróides (média das features para cada cluster)
    centroids_scaled = user_profiles_encoded.groupby('cluster')[musical_features].mean()

    # 2. Inverter o escalonamento para ter as características em sua escala original (ou pseudo-original)
    # Nota: Precisamos criar um DataFrame de centróides que inclua todas as colunas
    # que foram usadas no scaler original (X_features_modelo).
    # Como o scaler_modelo foi ajustado em X_scaled_modelo (que contém demográficos e musicais),
    # a inversão do escalonamento pode ser complexa.
    # Para simplificar e mostrar a DOMINÂNCIA, vamos focar nos valores ESCALONADOS,
    # que indicam o desvio padrão da média (0.0), que já é explicativo.

    # 3. Analisar Centróides (Escalonados)
    # Centróides com valores escalonados (aproximadamente a média das características)

    # Criar uma visualização (Gráfico de Barras) para os 3 primeiros clusters (para evitar sobrecarga)
    clusters_a_plotar = user_profiles_encoded['cluster'].unique()

    plt.figure(figsize=(15, 5 * min(3, len(clusters_a_plotar))))  # Plota os 3 primeiros

    for i, cluster_id in enumerate(clusters_a_plotar[:3]):
        # Seleciona o centróide escalonado
        centroid = centroids_scaled.loc[cluster_id]

        # Filtra apenas as características musicais de interesse
        musical_centroid = centroid[musical_features]

        # Calcula a relevância (distância absoluta da média 0.0)
        relevance = musical_centroid.abs().sort_values(ascending=False)
        top_features = relevance.head(top_n_features)

        # Plota os desvios
        plt.subplot(min(3, len(clusters_a_plotar)), 1, i + 1)

        # Cores: Positivo (acima da média) é azul, Negativo (abaixo da média) é vermelho
        colors = ['red' if musical_centroid.loc[f] < 0 else 'blue' for f in top_features.index]

        # Plota os valores do centróide (que representam o desvio padrão da média)
        plt.bar(top_features.index, musical_centroid.loc[top_features.index], color=colors)

        plt.title(f'Cluster {cluster_id}: {top_n_features} Características Musicais Dominantes (Desvio da Média Geral)',
                  fontsize=14)
        plt.ylabel('Desvio Padrão da Média (Escalonado)')
        plt.axhline(0, color='gray', linewidth=0.8)  # Linha da Média Geral (0.0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    plt.show()