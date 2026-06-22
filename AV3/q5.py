import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer

from q1 import load_data, preprocess_data
from q3 import build_discretized_credit_limit, build_kmeans_clusters, make_kmeans
from q4 import evaluate_dbscan


def get_normalizers():
    return {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "Log1p": None
    }


def compare_preprocessing_kmeans(X, normalizers, manual_elbow_k_by_normalizer, save_path, k_range=(2, 30)):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    star_handles_left  = []
    star_handles_right = []

    k_values = list(range(k_range[0], k_range[1] + 1))

    best_k_by_normalizer = {}

    for name, transformer in normalizers.items():

        if name == "Log1p":
            X_proc = np.log1p(X)
        else:
            X_proc = transformer.fit_transform(X)

        _dummy_fig, _dummy_ax = plt.subplots()
        model = make_kmeans()
        elbow = KElbowVisualizer(
            model, k=k_range, metric="distortion",
            timings=False, ax=_dummy_ax
        )
        elbow.fit(X_proc)
        plt.close(_dummy_fig)

        line = axes[0].plot(
            elbow.k_values_,
            elbow.k_scores_,
            marker="o",
            label=name
        )[0]
        color = line.get_color()

        manual_k = manual_elbow_k_by_normalizer[name]
        idx = list(elbow.k_values_).index(manual_k)
        axes[0].scatter(
            manual_k, elbow.k_scores_[idx],
            marker="*", s=400, color=color,
            edgecolor="black", linewidth=1.5, zorder=10
        )
        star_handles_left.append(
            plt.Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=color, markeredgecolor="black",
                       markersize=14, label=f"{name}  k={manual_k}")
        )

        sil_scores = np.array([
            silhouette_score(
                X_proc,
                make_kmeans(n_clusters=k).fit_predict(X_proc)
            )
            for k in k_values
        ])

        line_sil  = axes[1].plot(k_values, sil_scores, marker="o", label=name)[0]
        color_sil = line_sil.get_color()

        best_idx = int(np.argmax(sil_scores))
        best_k   = k_values[best_idx]
        best_k_by_normalizer[name] = best_k

        axes[1].scatter(
            best_k, sil_scores[best_idx],
            marker="*", s=400, color=color_sil,
            edgecolor="black", linewidth=1.5, zorder=10
        )
        star_handles_right.append(
            plt.Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=color_sil, markeredgecolor="black",
                       markersize=14, label=f"{name}  k={best_k}")
        )

    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Numero de Clusters (k)")
    axes[0].set_ylabel("Distortion / Inertia")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Numero de Clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.3)

    for ax, star_handles in zip(axes, [star_handles_left, star_handles_right]):
        line_handles, _ = ax.get_legend_handles_labels()
        separator = plt.Line2D([0], [0], color="none", label="─" * 21)

        ax.legend(
            handles=line_handles + [separator] + star_handles,
            title="Pre-processamento",
            loc="upper right",
            fontsize=15,
            title_fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return best_k_by_normalizer


def plot_scatter_by_group(clusters_by_name, x, y, x_label, y_label, save_path, titles=None):
    titles = titles or {}
    fig, axes = plt.subplots(1, len(clusters_by_name), figsize=(22, 5))

    for ax, (name, labels) in zip(axes, clusters_by_name.items()):
        ax.scatter(x, y, c=labels, cmap="viridis", alpha=0.6)
        ax.set_title(titles.get(name, name))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_contingency_by_group(clusters_by_name, reference, x_label, y_label, save_path, titles=None):
    titles = titles or {}
    fig, axes = plt.subplots(1, len(clusters_by_name), figsize=(22, 5))

    for ax, (name, labels) in zip(axes, clusters_by_name.items()):
        ct = pd.crosstab(labels, reference)
        sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False)
        ax.set_title(titles.get(name, name))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def compute_knn_distances_by_normalizer(X, normalizers, min_samples, elbow_idx_by_normalizer, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    k_distances_by_normalizer = {}

    for ax, (name, normalizer) in zip(axes, normalizers.items()):
        X_transformed = np.log1p(X) if normalizer is None else normalizer.fit_transform(X)

        nn = NearestNeighbors(n_neighbors=min_samples).fit(X_transformed)
        distances, _ = nn.kneighbors(X_transformed)
        k_dist = np.sort(distances[:, -1])
        k_distances_by_normalizer[name] = k_dist

        elbow_idx = elbow_idx_by_normalizer[name]

        ax.plot(k_dist)
        ax.scatter(elbow_idx, k_dist[elbow_idx], color="red", zorder=5,
                   label=f"Cotovelo (eps≈{k_dist[elbow_idx]:.2f})")
        ax.legend()
        ax.set_title(name)
        ax.set_xlabel("Pontos (ordenados)")
        ax.set_ylabel(f"Distância ao {min_samples}º vizinho")
        ax.grid(alpha=0.3)

    plt.suptitle("Gráfico de k-distância por normalização", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return k_distances_by_normalizer


def compare_dbscan_params_normalizations(X, normalizers, elbow_idx_by_normalizer, k_distances_by_normalizer, min_samples):
    all_results = []
    labels_by_normalizer = {}

    for name, normalizer in normalizers.items():
        X_transformed = np.log1p(X) if normalizer is None else normalizer.fit_transform(X)

        elbow_idx = elbow_idx_by_normalizer[name]
        eps       = k_distances_by_normalizer[name][elbow_idx]

        result_df, labels = evaluate_dbscan(
            X_transformed,
            eps=eps,
            min_samples=min_samples,
            sample_size=1000,
            random_state=42
        )

        result_df["normalizer"]    = name
        labels_by_normalizer[name] = labels
        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True), labels_by_normalizer


def pipeline():
    os.makedirs("image/q5", exist_ok=True)

    df = preprocess_data(load_data())
    X_cut = build_kmeans_clusters(df)
    X_features = X_cut.drop(columns=["cluster"])
    discrete_cols, _ = build_discretized_credit_limit(X_cut)
    col_to_discretize = 'CREDIT_LIMIT'

    normalizers = get_normalizers()

    manual_elbow_k_by_normalizer = {
        "StandardScaler": 7,
        "RobustScaler": 6,
        "MinMaxScaler": 3,
        "Log1p": 4,
    }

    compare_preprocessing_kmeans(
        X_features, normalizers, manual_elbow_k_by_normalizer,
        save_path="image/q5/elbow_silhouette_normalizacoes_kmeans.png"
    )

    best_k_by_normalizer = {
        "StandardScaler": 7,
        "RobustScaler": 6,
        "MinMaxScaler": 6,
        "Log1p": 3
    }

    clusters_by_normalizer = {}

    for name, transformer in normalizers.items():
        X_proc = np.log1p(X_features) if name == "Log1p" else transformer.fit_transform(X_features)

        model = make_kmeans(n_clusters=best_k_by_normalizer[name])

        clusters_by_normalizer[name] = pd.Series(model.fit_predict(X_proc), index=X_features.index)

    plot_scatter_by_group(
        clusters_by_normalizer,
        X_cut["PRC_FULL_PAYMENT"],
        X_cut["CREDIT_LIMIT"],
        "PRC_FULL_PAYMENT",
        "CREDIT_LIMIT",
        save_path="image/q5/scatter_kmeans_normalizacoes.png",
        titles={name: f"{name}  (k={best_k_by_normalizer[name]})" for name in clusters_by_normalizer}
    )

    plot_contingency_by_group(
        clusters_by_normalizer,
        discrete_cols[col_to_discretize],
        f"{col_to_discretize} (Sturges)",
        "Cluster",
        save_path="image/q5/crosstab_kmeans_normalizacoes.png",
    )

    X_cut_features = X_features
    min_samples = X_cut_features.shape[1] * 2

    dbscan_elbow_idx_by_normalizer = {
        "StandardScaler": 8000,
        "RobustScaler":   8000,
        "MinMaxScaler":   7500,
        "Log1p":          7000,
    }

    k_distances_by_normalizer = compute_knn_distances_by_normalizer(
        X_cut_features, normalizers, min_samples, dbscan_elbow_idx_by_normalizer,
        save_path="image/q5/kdistance_normalizacoes_dbscan.png"
    )

    dbscan_results_norm, dbscan_labels_by_normalizer = compare_dbscan_params_normalizations(
        X_cut_features, normalizers, dbscan_elbow_idx_by_normalizer, k_distances_by_normalizer, min_samples
    )

    print(dbscan_results_norm)

    plot_scatter_by_group(
        dbscan_labels_by_normalizer,
        X_cut["PRC_FULL_PAYMENT"],
        X_cut["CREDIT_LIMIT"],
        "PRC_FULL_PAYMENT",
        "CREDIT_LIMIT",
        save_path="image/q5/scatter_dbscan_normalizacoes.png",
        titles={
            name: f"{name}  ({labels[labels != -1].nunique()} clusters)"
            for name, labels in dbscan_labels_by_normalizer.items()
        }
    )

    plot_contingency_by_group(
        dbscan_labels_by_normalizer,
        discrete_cols[col_to_discretize],
        f"{col_to_discretize} (Sturges)",
        "Cluster (-1 = ruído)",
        save_path="image/q5/crosstab_dbscan_normalizacoes.png",
    )


if __name__ == "__main__":
    pipeline()
