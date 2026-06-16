import numpy as np


def compute_single_linkage(cluster1,cluster2):
    # Menor distância entre qualquer par de pontos (um de cada cluster)
    c1 = np.array(cluster1)
    c2 = np.array(cluster2)
    # np.linalg.norm calcula a distância euclidiana para todas as combinações
    distances = [np.linalg.norm(p1 - p2) for p1 in c1 for p2 in c2    ]
    return np.min(distances)

def compute_complete_linkage(cluster1, cluster2):
    # Maior distância entre qualquer par de pontos (um de cada cluster)
    c1 = np.array(cluster1)
    c2 = np.array(cluster2)
    distances = [np.linalg.norm(p1 - p2) for p1 in c1 for p2 in c2]
    return np.max(distances)

def compute_average_linkage(cluster1, cluster2):
    # Média de todas as distâncias entre os pares de pontos
    c1 = np.array(cluster1)
    c2 = np.array(cluster2)
    distances = [np.linalg.norm(p1 - p2) for p1 in c1 for p2 in c2]
    return np.mean(distances)

def compute_centroid_linkage(cluster1,cluster2):
    # Distância entre os centroides (médias geométricas) dos dois clusters
    c1 = np.array(cluster1)
    c2 = np.array(cluster2)
    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]] #x1 y1
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]] #x2 y2

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))



