import numpy as np


def _pairwise_distances(cluster1, cluster2):
    """Matriz de distâncias euclidianas entre todos os pares (a de A, b de B)."""
    c1 = np.array(cluster1, dtype=float)
    c2 = np.array(cluster2, dtype=float)
    # diferença com broadcasting: shape (len(c1), len(c2), 2)
    diff = c1[:, np.newaxis, :] - c2[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def compute_single_linkage(cluster1, cluster2):
    # menor distância entre um ponto de A e um ponto de B
    return _pairwise_distances(cluster1, cluster2).min()


def compute_complete_linkage(cluster1, cluster2):
    # maior distância entre um ponto de A e um ponto de B
    return _pairwise_distances(cluster1, cluster2).max()


def compute_average_linkage(cluster1, cluster2):
    # média de todas as distâncias par a par
    return _pairwise_distances(cluster1, cluster2).mean()


def compute_centroid_linkage(cluster1, cluster2):
    # distância entre os centroides dos dois clusters
    centroid1 = np.array(cluster1, dtype=float).mean(axis=0)
    centroid2 = np.array(cluster2, dtype=float).mean(axis=0)
    return np.sqrt(((centroid1 - centroid2) ** 2).sum())


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]  # x1 y1
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]  # x2 y2

print("similaridade ligação simples: ", compute_single_linkage(cluster1, cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1, cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1, cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1, cluster2))