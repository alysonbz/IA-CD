import numpy as np



# Distância Euclidiana
def euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



def compute_single_linkage(cluster1,cluster2):
    min_dist = float('inf')

    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidiana(p1, p2)
            if dist < min_dist:
                min_dist = dist

    return min_dist

def compute_complete_linkage(cluster1, cluster2):
    max_dist = 0

    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidiana(p1, p2)
            if dist > max_dist:
                max_dist = dist

    return max_dist

def compute_average_linkage(cluster1, cluster2):
    distances = []

    for p1 in cluster1:
        for p2 in cluster2:
            distances.append(euclidiana(p1, p2))

    return np.mean(distances)

def compute_centroid_linkage(cluster1,cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    return euclidiana(centroid1, centroid2)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]] #x1 y1
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]] #x2 y2

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))



