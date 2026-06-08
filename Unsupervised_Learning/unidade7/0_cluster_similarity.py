import numpy as np

def compute_single_linkage(cluster1,cluster2):
     links = []
     for i in range(len(cluster1)):
         for j in range(len(cluster2)):
             links.append(np.sqrt((cluster1[i][0]-cluster2[j][0])**2 + (cluster1[i][1]-cluster2[j][1])**2))
     return min(links)

def compute_complete_linkage(cluster1, cluster2):
    links = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            links.append(np.sqrt((cluster1[i][0] - cluster2[j][0]) ** 2 + (cluster1[i][1] - cluster2[j][1]) ** 2))
    return max(links)

def compute_average_linkage(cluster1, cluster2):
    links = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            links.append(np.sqrt((cluster1[i][0] - cluster2[j][0]) ** 2 + (cluster1[i][1] - cluster2[j][1]) ** 2))
    return np.mean(links)

def compute_centroid_linkage(cluster1,cluster2):
    x1,x2 = 0,0
    y1,y2 = 0,0
    for i in range(len(cluster1)):
        x1 += cluster1[i][0]
        y1 += cluster1[i][1]
        x2 += cluster2[i][0]
        y2 += cluster2[i][1]
    x1,y1 = x1/len(cluster1), y1/len(cluster1)
    x2,y2 = x2/len(cluster2), y2/len(cluster2)

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]] #x1 y1
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]] #x2 y2

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))



