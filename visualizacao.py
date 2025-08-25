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