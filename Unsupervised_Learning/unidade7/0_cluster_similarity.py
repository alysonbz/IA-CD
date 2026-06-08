import numpy as np

def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_single_linkage(cluster1, cluster2):
    distancias = []

    for p1 in cluster1:
        for p2 in cluster2:
            distancias.append(distancia(p1, p2))

    return min(distancias)

def compute_complete_linkage(cluster1, cluster2):
    distancias = []

    for p1 in cluster1:
        for p2 in cluster2:
            distancias.append(distancia(p1, p2))

    return max(distancias)

def compute_average_linkage(cluster1, cluster2):
    distancias = []

    for p1 in cluster1:
        for p2 in cluster2:
            distancias.append(distancia(p1, p2))

    return np.mean(distancias)

def compute_centroid_linkage(cluster1, cluster2):

    centroide1 = np.mean(cluster1, axis=0)
    centroide2 = np.mean(cluster2, axis=0)

    return distancia(centroide1, centroide2)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples:", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa:", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média:", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide:", compute_centroid_linkage(cluster1,cluster2))