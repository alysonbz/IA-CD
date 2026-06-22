from src.utils import load_wine_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd

df = load_wine_dataset()
X = df.drop(['class_label','class_name'], axis=1)
y = df['class_label']
print(y.value_counts())

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

ax[0].plot(range(1, 11), inertia, marker='o')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=3, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {3}")

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

ax[1].plot(range(2, 11), silhouette_scores, marker='o')
ax[1].set_title('Índice de Silhueta para Diferentes Valores de K')
ax[1].set_xlabel('Número de Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].axvline(x=2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {2}")
plt.show()

# Vamos usar 3 clusters pois é o número de dados alvos únicos

kmeans = KMeans(n_clusters=3, random_state=42)
scaler = StandardScaler()

pipe = make_pipeline(scaler, kmeans)
pipe.fit(X)
labels = pipe.predict(X)

df = pd.DataFrame({'Labels':labels,'Vinhos':y})
ct = pd.crosstab(df['Vinhos'], df['Labels'])
print(ct)

plt.scatter(X['alcohol'], X['alcalinity_of_ash'],c=y.values)
plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

X_ss = scaler.transform(X)
dendrogram(linkage(X_ss, method='ward'))
plt.show()

mergings = linkage(X_ss, method="ward")
labels = fcluster(mergings, 15, criterion="distance")

df = pd.DataFrame({'Labels':labels,'Vinhos':y})
ct = pd.crosstab(df['Vinhos'], df['Labels'])
print(ct)