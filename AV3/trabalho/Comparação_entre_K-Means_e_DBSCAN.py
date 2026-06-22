from dataset.pre_processamento import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

neighbors = NearestNeighbors(n_neighbors=10)
nbrs = neighbors.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances[:,9])

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].plot(distances)
ax[0].set_xlabel('Pontos ordenados')
ax[0].set_ylabel('Distância ao 10º vizinho')
ax[0].axhline(y=4200, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"eps escolhido = {4200}")

dbscan = DBSCAN(eps=4200, min_samples=21)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print('Sem normalização')
print("Clusters:", n_clusters)
print("Ruído:", n_noise)

ax[1].scatter(
    all_batch['feature_9'], all_batch['feature_100'],
    c=labels, alpha=0.7
)

ct = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct = pd.crosstab(ct['gas'],ct['cluster'],margins=True)

sns.heatmap(ct,ax=ax[2],annot=True,fmt='d',cmap='Blues')
ax[2].set_xlabel('Cluster DBScan')
ax[2].set_ylabel('Gás Real')
ax[2].set_title('Sem normalização')
plt.show()
print(ct)


neighbors = NearestNeighbors(n_neighbors=10)
nbrs = neighbors.fit(X_ss)
distances, indices = nbrs.kneighbors(X_ss)
distances = np.sort(distances[:,9])

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].plot(distances)
ax[0].set_xlabel('Pontos ordenados')
ax[0].set_ylabel('Distância ao 10º vizinho')
ax[0].axhline(y=4, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"eps escolhido = {4}")

dbscan = DBSCAN(eps=4, min_samples=21)
labels = dbscan.fit_predict(X_ss)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print('Standard Scaler')
print("Clusters:", n_clusters)
print("Ruído:", n_noise)

ax[1].scatter(
    all_batch['feature_9'], all_batch['feature_100'],
    c=labels, alpha=0.7
)

ct1 = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct1 = pd.crosstab(ct1['gas'],ct1['cluster'],margins=True)

sns.heatmap(ct1,ax=ax[2],annot=True,fmt='d',cmap='Blues')
ax[2].set_xlabel('Cluster DBScan')
ax[2].set_ylabel('Gás Real')
ax[2].set_title('Standard Scaler')
plt.show()


neighbors = NearestNeighbors(n_neighbors=10)
nbrs = neighbors.fit(X_mm)
distances, indices = nbrs.kneighbors(X_mm)
distances = np.sort(distances[:,9])

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].plot(distances)
ax[0].set_xlabel('Pontos ordenados')
ax[0].set_ylabel('Distância ao 10º vizinho')
ax[0].axhline(y=0.2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"eps escolhido = {0.2}")

dbscan = DBSCAN(eps=0.2, min_samples=21)
labels = dbscan.fit_predict(X_mm)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print('Min-Max Scaler')
print("Clusters:", n_clusters)
print("Ruído:", n_noise)

ax[1].scatter(
    all_batch['feature_9'], all_batch['feature_100'],
    c=labels, alpha=0.7
)

ct2 = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct2 = pd.crosstab(ct2['gas'],ct2['cluster'],margins=True)

sns.heatmap(ct2,ax=ax[2],annot=True,fmt='d',cmap='Blues')
ax[2].set_xlabel('Cluster DBScan')
ax[2].set_ylabel('Gás Real')
ax[2].set_title('MinMax Scaler')
plt.show()