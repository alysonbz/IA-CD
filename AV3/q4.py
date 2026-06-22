import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from q1 import load_data, preprocess_data
from q3 import build_discretized_credit_limit, build_kmeans_clusters, plot_clusters_scatter, plot_contingency


def compute_knn_distance_elbow(X, n_neighbors, elbow_idx, save_path):
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # Índice do cotovelo escolhido manualmente (olhando o gráfico abaixo); edite aqui pra mudar.
    elbow_value = k_distances[elbow_idx]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.scatter(elbow_idx, elbow_value, color="red", zorder=5, label=f"Cotovelo (eps={elbow_value:.2f})")
    plt.legend()
    plt.xlabel("Pontos (ordenados pela distância)")
    plt.ylabel(f"Distância ao {n_neighbors}º vizinho mais próximo")
    plt.title("Gráfico de k-distância - dados sem normalização")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close(fig)

    return elbow_value


def evaluate_dbscan(X, eps, min_samples, sample_size=None, random_state=None):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1

    score = np.nan
    if n_clusters >= 2:
        effective_sample_size = min(sample_size, mask.sum()) if sample_size is not None else None
        try:
            score = silhouette_score(
                X[mask], labels[mask],
                sample_size=effective_sample_size,
                random_state=random_state
            )
        except ValueError:
            pass

    result = pd.DataFrame([{
        "eps":        eps,
        "min_samples": min_samples,
        "clusters":   n_clusters,
        "noise_pct":  (labels == -1).mean() * 100,
        "silhouette": score
    }])

    labels_series = pd.Series(labels, index=X.index if hasattr(X, 'index') else None)

    return result, labels_series


def pipeline():
    os.makedirs("image/q4", exist_ok=True)

    df = preprocess_data(load_data())
    X_cut = build_kmeans_clusters(df)
    X_cut_features = X_cut.drop(columns=["cluster"])

    elbow_value = compute_knn_distance_elbow(
        X_cut_features, n_neighbors=18, elbow_idx=8100,
        save_path="image/q4/kdistance_sem_normalizacao.png"
    )

    min_samples = X_cut_features.shape[1] * 2

    dbscan_results_no_norm, dbscan_labels_no_norm = evaluate_dbscan(
        X_cut_features, eps=elbow_value, min_samples=min_samples
    )
    print(dbscan_results_no_norm)

    plot_clusters_scatter(
        X_cut["PRC_FULL_PAYMENT"],
        X_cut["CREDIT_LIMIT"],
        dbscan_labels_no_norm,
        "PRC_FULL_PAYMENT",
        "CREDIT_LIMIT",
        color_label="Cluster (-1 = ruído)",
        title=f"DBSCAN sem normalização",
        save_path="image/q4/scatter_dbscan_sem_normalizacao.png",
    )

    discrete_cols, _ = build_discretized_credit_limit(X_cut)
    col_to_discretize = 'CREDIT_LIMIT'

    ct_dbscan_no_norm = pd.crosstab(dbscan_labels_no_norm, discrete_cols[col_to_discretize])

    plot_contingency(
        ct_dbscan_no_norm,
        f"{col_to_discretize} (Sturges)",
        "Cluster (-1 = ruído)",
        f"DBSCAN sem normalização - Cluster vs {col_to_discretize}",
        save_path="image/q4/crosstab_dbscan_sem_normalizacao.png",
        figsize=(12, 4)
    )


if __name__ == "__main__":
    pipeline()
