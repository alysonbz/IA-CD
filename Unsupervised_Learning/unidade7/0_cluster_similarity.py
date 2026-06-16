import numpy as np


def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def pairwise_distances(c1, c2):
    return [euclidean(p1, p2) for p1 in c1 for p2 in c2]


def compute_single_linkage(c1, c2):
    return min(pairwise_distances(c1, c2))


def compute_complete_linkage(c1, c2):
    return max(pairwise_distances(c1, c2))


def compute_average_linkage(c1, c2):
    return np.mean(pairwise_distances(c1, c2))


def compute_centroid_linkage(c1, c2):
    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)
    return euclidean(centroid1, centroid2)


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

print("similaridade ligação simples:",
      compute_single_linkage(cluster1, cluster2))
print("similaridade ligação completa:",
      compute_complete_linkage(cluster1, cluster2))
print("similaridade ligação média:", compute_average_linkage(cluster1, cluster2))
print("similaridade pelo método do centroide:",
      compute_centroid_linkage(cluster1, cluster2))
