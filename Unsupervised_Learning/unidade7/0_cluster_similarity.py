import numpy as np


#Calcular a ligação simples
def compute_single_linkage(cluster1,cluster2):
    
    menor_distancia = float('inf')
    
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(
                np.array(ponto1) - np.array(ponto2)
            )
    
            if distancia < menor_distancia:
                menor_distancia = distancia
    
    return menor_distancia

#Calcular a ligação completa
def compute_complete_linkage(cluster1, cluster2):
    
    maior_distancia = 0
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(
                np.array(ponto1) - np.array(ponto2)
            )
        
            if distancia > maior_distancia:
                maior_distancia = distancia
    
    return maior_distancia

#Calcular a ligação média
def compute_average_linkage(cluster1, cluster2):
    
    soma_distancias = 0
    quantidade = 0
    
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(
                np.array(ponto1) - np.array(ponto2)
            )
            
            soma_distancias += distancia
            
            quantidade += 1
        

    return soma_distancias / quantidade

#Calcular o método do centroide
def compute_centroid_linkage(cluster1,cluster2):
    
    centroide1 = np.mean(cluster1, axis=0)
    
    centroide2 = np.mean(cluster2, axis=0)
    
    distancia = np.linalg.norm(centroide1 - centroide2)
    
    
    return distancia


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]] #x1 y1
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]] #x2 y2

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
