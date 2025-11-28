import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def print_playlist_bonita(playlist):
    if playlist.empty:
        print("🎧 Nenhuma música recomendada.")
        return

    cols = ['name', 'artists', 'popularity', 'artist_genres']
    # Adiciona 'origem' se existir
    if 'origem' in playlist.columns:
        print("passou")
        cols.append('origem')

    data = playlist[cols].copy()
    data['artist_genres'] = data['artist_genres'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    # Calcula larguras dinâmicas
    name_width = max(data['name'].str.len().max(), len("Nome")) + 2
    artist_width = max(data['artists'].str.len().max(), len("Artista(s)")) + 2
    pop_width = max(len(str(data['popularity'].max())), len("Popularidade")) + 2
    genres_width = max(data['artist_genres'].str.len().max(), len("Gêneros")) + 2
    origem_width = max(data['origem'].str.len().max(), len("Origem")) + 2 if 'origem' in data.columns else 0

    # Cabeçalho
    header = f"{'Nome'.ljust(name_width)}{'Artista(s)'.ljust(artist_width)}{'Popularidade'.ljust(pop_width)}{'Gêneros'.ljust(genres_width)}"
    if 'origem' in data.columns:
        header += f"{'Origem'.ljust(origem_width)}"

    print("\n🎧 Playlist Recomendada:\n")
    print(header)
    print("-" * len(header))

    # Linhas
    for idx, row in data.iterrows():
        linha = f"{row['name'].ljust(name_width)}{row['artists'].ljust(artist_width)}{str(row['popularity']).ljust(pop_width)}{row['artist_genres'].ljust(genres_width)}"
        if 'origem' in data.columns:
            linha += f"{row['origem'].ljust(origem_width)}"
        print(linha)

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


def plot_caracteristicas_comparativas_clusters(df_clusters_selecionados, musical_features, top_n_features=5,
                                               title="Características dos Clusters Atribuídos ao Usuário"):
    """
    Plota as características musicais mais dominantes de cada cluster atribuído.
    O DataFrame de entrada deve ter o ID do cluster no índice.
    """
    num_clusters = len(df_clusters_selecionados)

    # Define o tamanho da figura: 5 unidades de altura por cluster.
    plt.figure(figsize=(10, 5 * num_clusters))

    # 1. Plotar cada cluster separadamente
    for i, cluster_id in enumerate(df_clusters_selecionados.index):
        # O perfil para o cluster atual
        perfil = df_clusters_selecionados.loc[cluster_id]

        # Foca nas 5 características que mais se desviam de zero (média geral),
        # usando os valores ABSOLUTOS (escalonados) se for o caso,
        # mas como estamos usando centróides REVERTIDOS, focamos nos valores mais altos.
        top_features = perfil[musical_features].sort_values(ascending=False).head(top_n_features)

        # 2. Cria o subplot
        plt.subplot(num_clusters, 1, i + 1)

        # Usar os valores de volta à escala original (top_features.values)
        plt.bar(top_features.index, top_features.values, color='skyblue')

        plt.title(f'Cluster {cluster_id}: Top {top_n_features} Características Musicais Dominantes', fontsize=14)
        plt.ylabel('Valor Médio do Atributo')
        plt.xlabel('Características Musicais')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

    # Título geral para a figura (se necessário)
    plt.suptitle(title, y=1.02, fontsize=16)

    plt.show()

def gerar_plots_clusters(user_profiles_encoded, musical_features, genero_col="gender", genero_musical_col="main_genre"):
    """
    Gera plots completos para validar os clusters:
    - Distribuição de gênero (sexo) por cluster
    - Atributos musicais dominantes por cluster
    - Gênero musical dominante por cluster
    - Visualização PCA dos clusters (opcional)
    """

    # ----------------------------------------------------------------------
    # 1. PLOT — DISTRIBUIÇÃO DE GÊNERO POR CLUSTER
    # ----------------------------------------------------------------------
    print("\n[Plot] Distribuição de gênero (sexo) por cluster...")

    if genero_col in user_profiles_encoded.columns:
        # caso seja categórico simples: 'gender'
        gender_counts = user_profiles_encoded.groupby('cluster')[genero_col].value_counts().unstack().fillna(0)

        plt.figure(figsize=(8, 5))
        gender_counts.plot(kind='bar', stacked=True)
        plt.title('Distribuição de Gênero por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Número de Usuários')
        plt.legend(title='Gênero')
        plt.tight_layout()
        plt.show()

    else:
        # caso tenha sido convertido para one-hot
        gender_cols = [c for c in user_profiles_encoded.columns if c.startswith("gender_")]
        if len(gender_cols) > 0:
            gender_means = user_profiles_encoded.groupby('cluster')[gender_cols].mean()

            plt.figure(figsize=(8, 5))
            gender_means.plot(kind='bar', stacked=True)
            plt.title('Proporção de Gêneros por Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Proporção')
            plt.tight_layout()
            plt.show()

    # ----------------------------------------------------------------------
    # 2. PLOT — MÉDIA DOS ATRIBUTOS MUSICAIS POR CLUSTER
    # ----------------------------------------------------------------------
    print("\n[Plot] Atributos musicais dominantes por cluster...")

    music_feats_existentes = [f for f in musical_features if f in user_profiles_encoded.columns]

    means_music = user_profiles_encoded.groupby('cluster')[music_feats_existentes].mean()

    plt.figure(figsize=(10, 6))
    means_music.plot(kind='bar')
    plt.title('Médias dos Atributos Musicais por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Valor Médio')
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------
    # 3. PLOT — DISTRIBUIÇÃO DOS GÊNEROS MUSICAIS POR CLUSTER
    # ----------------------------------------------------------------------
    print("\n[Plot] Gênero musical dominante por cluster...")

    if genero_musical_col in user_profiles_encoded.columns:
        # gênero musical categórico simples
        genre_counts = user_profiles_encoded.groupby('cluster')[genero_musical_col].value_counts().unstack().fillna(0)

        plt.figure(figsize=(12, 7))
        genre_counts.plot(kind='bar', stacked=True)
        plt.title('Distribuição dos Gêneros Musicais por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.show()

    else:
        # caso seja one-hot encoding dos gêneros
        genre_cols = [c for c in user_profiles_encoded.columns if c.startswith("genre_")]
        if len(genre_cols) > 0:
            genre_means = user_profiles_encoded.groupby('cluster')[genre_cols].mean()

            plt.figure(figsize=(12, 6))
            genre_means.plot(kind='bar', stacked=True)
            plt.title('Proporção de Gêneros Musicais por Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Proporção')
            plt.tight_layout()
            plt.show()

    print("\n✔️ Todos os plots foram gerados com sucesso.")