import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    # --- Radar Chart para cada cluster ---
    labels = np.array(musical_features)
    num_vars = len(labels)

    # Ângulos para o gráfico circular
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # fecha o círculo

    plt.figure(figsize=(8, 8))
    for cluster_id, row in cluster_means.iterrows():
        values = row.tolist()
        values += values[:1]
        plt.polar(angles, values, label=f"Cluster {cluster_id}")
        plt.fill(angles, values, alpha=0.1)

    plt.title("Perfis médios por cluster (Radar Chart)")
    plt.xticks(angles[:-1], labels, fontsize=10)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.show()
