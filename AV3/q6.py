import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

from q1 import load_data, preprocess_data
from q3 import build_kmeans_clusters, make_kmeans, plot_clusters_scatter


def scale_features(X):
    return RobustScaler().fit_transform(X)


def build_hierarchical_clusters(X_scaled, methods_cuts, save_dir=None):
    hierarchical_labels = {}
    for metd, cut_height in methods_cuts.items():

        Z = linkage(X_scaled, method=metd, metric="euclidean")

        labels = fcluster(Z, t=cut_height, criterion="distance") - 1  # 0-based
        n_clusters_suggested = len(np.unique(labels))
        hierarchical_labels[f"hierarchical_{metd}"] = labels          # << salva

        if save_dir is not None:
            fig = plt.figure(figsize=(14, 6))
            dendrogram(Z, truncate_mode="lastp", p=50, show_leaf_counts=True, leaf_rotation=90)
            plt.axhline(y=cut_height, color="red", linestyle="--",
                        label=f"Corte em {cut_height} ({n_clusters_suggested} clusters)")
            plt.legend()
            plt.title(f"Dendrograma - {metd} (últimas 50 uniões)")
            plt.ylabel("Distância")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/dendrograma_{metd}.png")
            plt.close(fig)

    return hierarchical_labels


def build_target_labels(X_scaled, hierarchical_labels, verbose=False):
    cluster_target_models = {
        "agglomerative_ward": AgglomerativeClustering(n_clusters=4, linkage="ward"),
        "kmeans": make_kmeans(n_clusters=2),
        **hierarchical_labels,
    }

    TARGET_MODEL = "hierarchical_ward"   # troque aqui para usar outro método

    target_source = cluster_target_models[TARGET_MODEL]

    if isinstance(target_source, np.ndarray):
        target_labels = target_source
    else:
        target_labels = target_source.fit_predict(X_scaled)

    if verbose:
        print(pd.Series(target_labels).value_counts())

    return target_labels


def get_target_labels(df):
    X_cut = build_kmeans_clusters(df)
    X_cut_features = X_cut.drop(columns=["cluster"])
    X_cut_scaled = scale_features(X_cut_features)

    methods_cuts = {"single": 15, "average": 35, "ward": 200, "complete": 40}
    hierarchical_labels = build_hierarchical_clusters(X_cut_scaled, methods_cuts, save_dir=None)
    target_labels = build_target_labels(X_cut_scaled, hierarchical_labels, verbose=False)

    return target_labels, X_cut_features


def pipeline():
    os.makedirs("image/q6", exist_ok=True)

    df = preprocess_data(load_data())
    X_cut = build_kmeans_clusters(df)
    X_cut_features = X_cut.drop(columns=["cluster"])
    X_cut_scaled = scale_features(X_cut_features)

    methods_cuts = {"single": 15, "average": 35, "ward": 200, "complete": 40}
    hierarchical_labels = build_hierarchical_clusters(X_cut_scaled, methods_cuts, save_dir="image/q6")

    target_labels = build_target_labels(X_cut_scaled, hierarchical_labels, verbose=True)

    plot_clusters_scatter(
        X_cut["PRC_FULL_PAYMENT"],
        X_cut["CREDIT_LIMIT"],
        target_labels,
        "PRC_FULL_PAYMENT",
        "CREDIT_LIMIT",
        title=f"Clusters (hierarchical_ward)",
        save_path="image/q6/scatter_clusters_hierarchical_ward.png",
    )

    return target_labels, X_cut_features


if __name__ == "__main__":
    pipeline()
