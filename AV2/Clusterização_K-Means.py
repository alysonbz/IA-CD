# ===========================================
# Clusterização K-Means e Hierárquica
# ===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# 1. Preparação dos dados
# -----------------------------
df = pd.read_csv("StudentsPerformance.csv")

# Selecionar variáveis numéricas
X = df[['math score', 'reading score', 'writing score']]

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE para visualização
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

# -----------------------------
# 2. K-Means: escolher k
# -----------------------------
inertia = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualizar Elbow e Silhouette
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inertia', color=color)
ax1.plot(K_range, inertia, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(K_range, silhouette_scores, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means: Elbow Method e Silhouette Score')
plt.show()

# Melhor k pelo silhouette
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Melhor número de clusters: {best_k}")

# K-Means final
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

# Visualização PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, palette='Set1')
plt.title('K-Means Clusters (PCA)')
plt.show()

# Visualização t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_kmeans, palette='Set1')
plt.title('K-Means Clusters (t-SNE)')
plt.show()

# -----------------------------
# 3. Clusterização Hierárquica
# -----------------------------
linkage_methods = ['average', 'complete']
silhouette_hier = {}

for method in linkage_methods:
    hier = AgglomerativeClustering(n_clusters=best_k, linkage=method)
    labels_hier = hier.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_hier)
    silhouette_hier[method] = score

    # PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_hier, palette='Set2')
    plt.title(f'Hierarchical ({method}) - PCA')
    plt.show()

    # t-SNE
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_hier, palette='Set2')
    plt.title(f'Hierarchical ({method}) - t-SNE')
    plt.show()

# Dendrograma para análise qualitativa
plt.figure(figsize=(10, 6))
Z = linkage(X_scaled, method='average')
dendrogram(Z)
plt.title('Dendrograma (Average Linkage)')
plt.show()

# -----------------------------
# Comparação de métricas
# -----------------------------
print("Silhouette Scores:")
print(f"K-Means: {max(silhouette_scores):.3f}")
for method, score in silhouette_hier.items():
    print(f"Hierárquico ({method}): {score:.3f}")
