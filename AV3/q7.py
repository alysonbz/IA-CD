import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from q1 import load_data, preprocess_data
from q6 import get_target_labels


def plot_top_pca_components(X, pca, save_path, n_components=None, top_n=10):
    X_scaled = RobustScaler().fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = n_components
        pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_

    result_df = pd.DataFrame({
        "Componente": [f"PC{i}" for i in range(1, len(explained_var) + 1)],
        "Variancia_Explicada": explained_var,
        "Variancia_Acumulada": np.cumsum(explained_var)
    })
    result_df = result_df.sort_values("Variancia_Explicada", ascending=False).head(top_n)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(result_df["Componente"], result_df["Variancia_Explicada"])
    plt.xlabel("Componente Principal")
    plt.ylabel("Variância Explicada")
    plt.title(f"Top {top_n} Componentes Mais Importantes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return result_df


def plot_pca_3d(X, pca, save_path, y=None, scaler=None):
    scaler = scaler or RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = 3
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)[:, :3]

    var = pca.explained_variance_ratio_[:3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, alpha=0.7, cmap="tab10")
        plt.colorbar(scatter)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var[2]:.1%})")
    ax.set_title(f"PCA 3D - Variância acumulada: {var[:3].sum():.1%}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return X_pca, pca


def plot_pca_2d(X, pca, save_path, y=None, scaler=None):
    scaler = scaler or RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = 2
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)[:, :2]

    var = pca.explained_variance_ratio_[:2]

    fig, ax = plt.subplots(figsize=(10, 7))

    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7, cmap="tab10")
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title(f"PCA 2D - Variância acumulada: {var[:2].sum():.1%}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return X_pca


def run_knn_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf):
    knn_normalizers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "Log1p": None
    }

    knn_results = []
    knn_confusion = {}
    knn_best_results = []

    for norm_name, normalizer in knn_normalizers.items():
        if normalizer is None:
            X_train_proc = np.log1p(X_train_clf)
            X_test_proc = np.log1p(X_test_clf)
        else:
            X_train_proc = normalizer.fit_transform(X_train_clf)
            X_test_proc = normalizer.transform(X_test_clf)

        pca4 = PCA(n_components=4, random_state=77)
        X_train_pca = pca4.fit_transform(X_train_proc)
        X_test_pca = pca4.transform(X_test_proc)

        feature_sets = {
            "Todas as colunas": (X_train_proc, X_test_proc),
            "Top 4 PCA": (X_train_pca, X_test_pca),
        }

        for feature_set_name, (X_tr, X_te) in feature_sets.items():
            best_for_combo = None

            for n_neighbors in range(2, 21):
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(X_tr, y_train_clf)
                y_pred = knn.predict(X_te)

                report = classification_report(y_test_clf, y_pred, output_dict=True)
                acc = report["accuracy"]

                knn_results.append({
                    "normalizacao": norm_name,
                    "features": feature_set_name,
                    "n_neighbors": n_neighbors,
                    "accuracy": acc,
                    "precision": report["weighted avg"]["precision"],
                    "recall": report["weighted avg"]["recall"],
                    "f1": report["weighted avg"]["f1-score"],
                })

                if best_for_combo is None or acc > best_for_combo["accuracy"]:
                    best_for_combo = {
                        "n_neighbors": n_neighbors,
                        "accuracy": acc,
                        "y_pred": y_pred,
                        "report": report,
                    }

            knn_confusion[(norm_name, feature_set_name)] = confusion_matrix(
                y_test_clf, best_for_combo["y_pred"]
            )

            knn_best_results.append({
                "normalizacao": norm_name,
                "features": feature_set_name,
                "n_neighbors": best_for_combo["n_neighbors"],
                "accuracy": best_for_combo["accuracy"],
                "precision": best_for_combo["report"]["weighted avg"]["precision"],
                "recall": best_for_combo["report"]["weighted avg"]["recall"],
                "f1": best_for_combo["report"]["weighted avg"]["f1-score"],
            })

    norm_order = ["StandardScaler", "MinMaxScaler", "Log1p"]
    features_order = ["Todas as colunas", "Top 4 PCA"]

    knn_best_results_df = pd.DataFrame(knn_best_results)
    knn_best_results_df["normalizacao"] = pd.Categorical(
        knn_best_results_df["normalizacao"], categories=norm_order, ordered=True
    )
    knn_best_results_df["features"] = pd.Categorical(
        knn_best_results_df["features"], categories=features_order, ordered=True
    )
    knn_best_results_df = knn_best_results_df.sort_values(["normalizacao", "features"]).reset_index(drop=True)

    return knn_best_results_df, knn_confusion


def plot_knn_confusion_matrices(knn_confusion, save_path):
    feature_set_names = ["Todas as colunas", "Top 4 PCA"]
    norm_names = ["StandardScaler", "MinMaxScaler", "Log1p"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, feature_set_name in enumerate(feature_set_names):
        for col, norm_name in enumerate(norm_names):
            ax = axes[row, col]
            cm = knn_confusion[(norm_name, feature_set_name)]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{norm_name}\n{feature_set_name}")
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def run_tree_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf, save_dir):
    scaler_for_tree_pca = StandardScaler()
    X_train_tree_scaled = scaler_for_tree_pca.fit_transform(X_train_clf)
    X_test_tree_scaled = scaler_for_tree_pca.transform(X_test_clf)

    pca4_tree = PCA(n_components=4, random_state=77)
    X_train_tree_pca = pca4_tree.fit_transform(X_train_tree_scaled)
    X_test_tree_pca = pca4_tree.transform(X_test_tree_scaled)

    tree_feature_sets = {
        "Todas as colunas": (X_train_clf, X_test_clf),
        "Top 4 PCA": (X_train_tree_pca, X_test_tree_pca),
    }

    tree_results = []
    tree_models = {}

    file_suffix = {"Todas as colunas": "todas_colunas", "Top 4 PCA": "top4_pca"}

    for feature_set_name, (X_tr, X_te) in tree_feature_sets.items():
        tree_pipeline = Pipeline([
            ("feature_selection", VarianceThreshold(threshold=0.01)),
            ("tree", DecisionTreeClassifier(random_state=77)), # Critério padrão é o Gini
        ])

        tree_pipeline.fit(X_tr, y_train_clf)
        y_pred_tree = tree_pipeline.predict(X_te)

        print(f"=== Árvore de Decisão | {feature_set_name} ===")
        print(classification_report(y_test_clf, y_pred_tree))

        report = classification_report(y_test_clf, y_pred_tree, output_dict=True)
        tree_cm = confusion_matrix(y_test_clf, y_pred_tree)
        tree_models[feature_set_name] = tree_pipeline

        tree_results.append({
            "features": feature_set_name,
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        })

        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(tree_cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title(f"Árvore de Decisão - Matriz de Confusão ({feature_set_name})")
        plt.savefig(f"{save_dir}/matriz_confusao_arvore_{file_suffix[feature_set_name]}.png")
        plt.close(fig)

    tree_results_df = pd.DataFrame(tree_results).sort_values("f1", ascending=False).reset_index(drop=True)
    return tree_results_df, tree_models


def pipeline():
    os.makedirs("image/q7", exist_ok=True)

    df = preprocess_data(load_data())
    target_labels, X_cut_features = get_target_labels(df)

    shared_pca = PCA(random_state=77)

    result_df = plot_top_pca_components(
        X_cut_features, pca=shared_pca, top_n=10, save_path="image/q7/top_pca_componentes.png"
    )
    print(result_df)

    X_pca_3d, pca_3d = plot_pca_3d(
        X_cut_features, pca=shared_pca, y=target_labels, save_path="image/q7/pca_3d.png"
    )
    print((X_pca_3d, pca_3d))

    X_pca_2d = plot_pca_2d(
        X_cut_features, pca=shared_pca, y=target_labels, save_path="image/q7/pca_2d.png"
    )
    print(X_pca_2d)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_cut_features, target_labels,
        test_size=0.2, random_state=77, stratify=target_labels
    )

    knn_best_results_df, knn_confusion = run_knn_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    print(knn_best_results_df)

    plot_knn_confusion_matrices(knn_confusion, save_path="image/q7/matrizes_confusao_knn.png")

    tree_results_df, tree_models = run_tree_classification(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf, save_dir="image/q7"
    )
    print(tree_results_df)


if __name__ == "__main__":
    pipeline()
