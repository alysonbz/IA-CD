#Clusterização com K-Means e Hierárquico

#importando as bibliotecas necessárias
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

#Importando o csv com o pandas
df = pd.read_csv('mall_ajustado.csv')
print("________________________________________________________")

pca_model = PCA(n_components=2)
pca_features = pca_model.fit_transform(df)

#Usando os dados do PCA para visualização e clusterização
X = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])

#Testando vários k
inertias = []
silhouettes = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

#Plotando o metodo do cotovelo
plt.plot(K, inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')
plt.title('Método do Cotovelo - KMeans')
plt.show()

#Plotando o Silhouette Score
plt.plot(K, silhouettes, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Avaliação de Cluster - KMeans')
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# Visualização
plt.scatter(X['PC1'], X['PC2'], c=labels_kmeans, cmap='Set2')
plt.title('K-Means Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Ligação average
link_avg = linkage(X, method='average')
plt.figure(figsize=(10, 5))
dendrogram(link_avg)
plt.title('Dendrograma - Ligação Average')
plt.show()

#Ligação complete
link_complete = linkage(X, method='complete')
plt.figure(figsize=(10, 5))
dendrogram(link_complete)
plt.title('Dendrograma - Ligação Complete')
plt.show()

#Aplicar Agglomerative Clustering
agg_avg = AgglomerativeClustering(n_clusters=4, linkage='average')
labels_avg = agg_avg.fit_predict(X)

agg_comp = AgglomerativeClustering(n_clusters=4, linkage='complete')
labels_comp = agg_comp.fit_predict(X)

#Visualização
plt.scatter(X['PC1'], X['PC2'], c=labels_avg, cmap='Set1')
plt.title('Hierárquico (Average) - PCA')
plt.show()

plt.scatter(X['PC1'], X['PC2'], c=labels_comp, cmap='Set3')
plt.title('Hierárquico (Complete) - PCA')
plt.show()

df['cluster'] = labels_kmeans
df.to_csv('mall_ajustado.csv', index=False)

# Comparação dos métodos:
# - O K-Means formou grupos mais compactos, com Silhouette Score de X.XX
# - O Hierárquico (average) mostrou agrupamentos semelhantes, mas com menos separação visual.
# - A escolha ideal pode depender do objetivo: para agrupamentos rápidos e bem definidos, K-Means funciona melhor. Já o Hierárquico é útil para análises exploratórias e dendrogramas.
