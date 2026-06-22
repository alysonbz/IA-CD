import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from q1 import load_data, preprocess_data


def plot_correlation_heatmap(df, save_path):
    fig = plt.figure(figsize=(14, 10))
    corr = df.corr(method="spearman", numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
    plt.savefig(save_path)
    plt.close(fig)


def show_correlated_pairs(X, threshold=0.80):
    corr = X.corr(method="spearman", numeric_only=True)

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    pairs = []

    for col in upper.columns:
        for row in upper.index:
            value = upper.loc[row, col]

            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((abs(value), row, col, value))

    pairs.sort(reverse=True)

    print(f"Pares com |corr| >= {threshold}:\n")

    for _, var1, var2, corr_value in pairs:
        print(f"[{corr_value:.3f}] {var1} <--> {var2}")

    return pairs


def select_features(df):
    vars_remover = [
        'CASH_ADVANCE_TRX',                   # contagem bruta → quase idêntica a CASH_ADVANCE_FREQUENCY (corr 0.98)
        'CASH_ADVANCE_FREQUENCY',             # frequência → substituída por CASH_ADVANCE (valor monetário mais informativo pra risco)
        'MINIMUM_PAYMENTS',                   # derivada → substituída por BALANCE
        'PURCHASES_TRX',                      # contagem bruta → substituída por PURCHASES
        'ONEOFF_PURCHASES',                   # subcomponente → capturado por PURCHASES
        'PURCHASES_FREQUENCY',                # frequência → optou-se pelo montante PURCHASES
        'INSTALLMENTS_PURCHASES',             # subcomponente → capturado por PURCHASES
        'PURCHASES_INSTALLMENTS_FREQUENCY',   # freq. parcelamento → removida em cascata
    ]

    X_cut = df.drop(columns=vars_remover)
    return X_cut


def make_kmeans(random_state=77, n_init=10, **kwargs):
    return KMeans(init="k-means++", random_state=random_state, n_init=n_init, **kwargs)


def _override_elbow_line(ax, k, titulo):
    """Remove a linha vertical automática do yellowbrick e insere a manual."""
    for line in list(ax.lines):
        xdata = np.array(line.get_xdata())
        if len(xdata) == 2 and xdata[0] == xdata[1]:  # linha vertical
            line.remove()

    ax.axvline(x=k, color='tomato', linestyle='--', linewidth=2, label=f'k = {k} (manual)')
    ax.set_title(titulo)
    ax.legend()


def get_silhoete_and_elbow(data, save_path, k_range=(2, 30), elbow_k=None, silhouette_k=None):

    model = make_kmeans()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    elbow = KElbowVisualizer(
        model, k=k_range, metric='distortion', timings=False, ax=axes[0]
    )
    elbow.fit(data)
    elbow.finalize()

    if elbow_k is not None:
        _override_elbow_line(axes[0], elbow_k, f'Distortion Elbow')

    silhouette = KElbowVisualizer(
        model, k=k_range, metric='silhouette', timings=False, ax=axes[1]
    )
    silhouette.fit(data)
    silhouette.finalize()

    if silhouette_k is not None:
        _override_elbow_line(axes[1], silhouette_k, f'Silhouette Score')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_clusters_scatter(x, y, labels, x_label, y_label, save_path, color_label="Cluster", title=None):
    fig = plt.figure()
    plt.scatter(x, y, c=labels, cmap="viridis", alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(label=color_label)
    if title:
        plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def plot_contingency(ct, x_label, y_label, title, save_path, figsize=(8, 4)):
    fig = plt.figure(figsize=figsize)
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def discretize_sturges(X, feature):
    """Discretiza usando a regra de Sturges: k = ceil(log2(n) + 1) bins de largura igual."""
    n = len(X)
    k_bins = int(np.ceil(np.log2(n) + 1))

    data_min, data_max = X[feature].min(), X[feature].max()
    bins = np.linspace(data_min, data_max, k_bins + 1)
    labels = list(range(k_bins))

    discretized = pd.cut(X[feature], bins=bins, include_lowest=True, labels=labels).astype(int)
    return discretized, k_bins


def build_kmeans_clusters(df):
    X_cut = select_features(df)
    model = make_kmeans(random_state=67, n_clusters=3)
    model.fit(X_cut)

    X_cut = X_cut.copy()
    X_cut["cluster"] = model.predict(X_cut)
    return X_cut


def build_discretized_credit_limit(X_cut):
    col_to_discretize = 'CREDIT_LIMIT'
    discrete_cols = pd.DataFrame()

    discrete_cols[col_to_discretize], k_bins = discretize_sturges(X_cut, col_to_discretize)
    return discrete_cols, k_bins


def pipeline():
    os.makedirs("image/q3", exist_ok=True)

    df = preprocess_data(load_data())

    plot_correlation_heatmap(df, save_path="image/q3/heatmap_correlacao_spearman.png")
    show_correlated_pairs(df, threshold=0.75)

    X_cut = select_features(df)

    get_silhoete_and_elbow(
        X_cut, save_path="image/q3/elbow_silhouette_k.png", elbow_k=5, silhouette_k=2
    )

    model = make_kmeans(random_state=67, n_clusters=3)
    model.fit(X_cut)

    X_cut["cluster"] = model.predict(X_cut)

    plot_clusters_scatter(
        X_cut["PRC_FULL_PAYMENT"],
        X_cut["CREDIT_LIMIT"],
        X_cut["cluster"],
        "PRC_FULL_PAYMENT",
        "CREDIT_LIMIT",
        save_path="image/q3/scatter_clusters_kmeans.png",
    )

    col_to_discretize = 'CREDIT_LIMIT'
    discrete_cols, k_bins = build_discretized_credit_limit(X_cut)
    print(f"{col_to_discretize} | Regra de Sturges: k = {k_bins} bins")

    ct = pd.crosstab(
        X_cut["cluster"],
        discrete_cols[col_to_discretize]
    )

    plot_contingency(
        ct, f"{col_to_discretize} (Sturges)", "Cluster", f"Cluster vs {col_to_discretize}",
        save_path="image/q3/crosstab_cluster_credit_limit.png", figsize=(12, 4)
    )

    return X_cut, discrete_cols


if __name__ == "__main__":
    pipeline()
