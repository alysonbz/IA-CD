import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA

# Dados ja padronizados
# 1. K-Means: Gráfico do Cotovelo (Inertia)
inertia = []
silhouette_scores = []
K_RANGE = range(2, 10)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_RANGE, inertia, marker='o')
plt.title("Gráfico do Cotovelo (Inertia)")
plt.xlabel("Número de Clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(K_RANGE, silhouette_scores, marker='o', color='orange')
plt.title("Silhouette Score para K-Means")
plt.xlabel("Número de Clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("kmeans_inertia_silhouette.png")
plt.show()

# Escolher k com melhor silhouette score para kmeans
k_best = K_RANGE[silhouette_scores.index(max(silhouette_scores))]
print(f"K ideal pelo Silhouette Score: {k_best}")

# Rodar K-Means com k_best
km = KMeans(n_clusters=k_best, random_state=42)
labels_km = km.fit_predict(X_scaled)

# 2. Clusterização Hierárquica (average e complete)
linked_avg = linkage(X_scaled, method='average')
linked_complete = linkage(X_scaled, method='complete')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
dendrogram(linked_avg, truncate_mode='lastp', p=12)
plt.title("Dendrograma - Average")
plt.subplot(1, 2, 2)
dendrogram(linked_complete, truncate_mode='lastp', p=12)
plt.title("Dendrograma - Complete")
plt.tight_layout()
plt.savefig("dendrogramas_average_complete.png")
plt.show()

# Definir número de clusters hierárquicos igual ao k_best para comparação
labels_hier_avg = fcluster(linked_avg, t=k_best, criterion='maxclust')
labels_hier_complete = fcluster(linked_complete, t=k_best, criterion='maxclust')

# Calcular silhouette score para hierárquico average e complete
sil_avg = silhouette_score(X_scaled, labels_hier_avg)
sil_complete = silhouette_score(X_scaled, labels_hier_complete)
print(f"Silhouette Hierárquico Average: {sil_avg:.4f}")
print(f"Silhouette Hierárquico Complete: {sil_complete:.4f}")

# 3. Visualização dos clusters com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='viridis', alpha=0.6)
plt.title(f"K-Means (k={k_best})\nSilhouette: {max(silhouette_scores):.2f}")

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hier_avg, cmap='plasma', alpha=0.6)
plt.title(f"Hierárquico Average (k={k_best})\nSilhouette: {sil_avg:.2f}")

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hier_complete, cmap='magma', alpha=0.6)
plt.title(f"Hierárquico Complete (k={k_best})\nSilhouette: {sil_complete:.2f}")

plt.tight_layout()
plt.savefig("comparacao_cluster_pca.png")
plt.show()
