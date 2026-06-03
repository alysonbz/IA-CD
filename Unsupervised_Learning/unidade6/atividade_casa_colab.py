# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

iris = load_iris()
data_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
data_iris.head(10)


# Método do Cotovelo
X = data_iris.values
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list(K_range), inertias, 'o-', color='green', linewidth=2, markersize=6)
plt.axvline(x=3, color='red', linestyle='--', alpha=0.6, label='Cotovelo em K=3')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.title('Método do Cotovelo — Dataset Iris')
plt.legend()
plt.tight_layout()
plt.show()

# Índice de Silhueta
sil_scores = []
K_range2 = range(2, 11)

for k in K_range2:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

melhor_k = list(K_range2)[sil_scores.index(max(sil_scores))]

plt.figure(figsize=(8, 4))
plt.plot(list(K_range2), sil_scores, 'o-', color='steelblue', linewidth=2, markersize=6)
plt.axvline(x=melhor_k, color='red', linestyle='--', alpha=0.6, label=f'Melhor K={melhor_k}')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Índice de Silhueta — Dataset Iris')
plt.legend()
plt.tight_layout()
plt.show()

print("PERGUNTA 1.1 — K ideal pelo método do cotovelo:")
print("  R: K = 3")
print("  No gráfico, a curva 'dobra' em K=3, indicando que")
print("  adicionar mais clusters além disso pouco melhora.")
print()
print("PERGUNTA 1.2 — Os métodos concordam?")
print("  R: Não. O cotovelo indica K=3 e a silhueta indica K=2.")
print("  A diferença ocorre porque duas espécies do Iris são")
print("  muito parecidas entre si, então a silhueta prefere")
print("  juntá-las em um único grupo.")