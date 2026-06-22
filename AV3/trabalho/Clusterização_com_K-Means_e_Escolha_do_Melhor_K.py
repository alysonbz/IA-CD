from dataset.pre_processamento import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

ax[0].plot(range(1, 11), inertia, marker='o')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {2}")
ax[0].axvline(x=3, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {3}")

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
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

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X)
labels = kmeans.predict(X)


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    ax[0].scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
ax[0].set_xlabel('feature_9')
ax[0].set_ylabel('feature_100')
ax[0].set_title('Standard Scaler')
ax[0].legend()

ct = pd.DataFrame({'label':labels,
                    'gases': y['gas'].values})
ct = pd.crosstab(ct['gases'],ct['label'], margins=True)
print(ct)

sns.heatmap(ct,ax=ax[1],annot=True,fmt='d',cmap='Blues')
ax[1].set_xlabel('Cluster K-Means')
ax[1].set_ylabel('Gás Real')
ax[1].set_title('Standard Scaler')
plt.show()

clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    plt.scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
plt.xlabel('feature_9')
plt.ylabel('feature_100')
plt.title('K-Means (K=6)')
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_ss)
    inertia.append(kmeans.inertia_)

ax[0].plot(range(1, 11), inertia, marker='o')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=3, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {3}")

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_ss)
    score = silhouette_score(X_ss, kmeans.labels_)
    silhouette_scores.append(score)

ax[1].plot(range(2, 11), silhouette_scores, marker='o')
ax[1].set_title('Índice de Silhueta para Diferentes Valores de K')
ax[1].set_xlabel('Número de Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].axvline(x=2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {2}")
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_ss)
labels = kmeans.predict(X_ss)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    ax[0].scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
ax[0].set_xlabel('feature_9')
ax[0].set_ylabel('feature_100')
ax[0].set_title('Standard Scaler')
ax[0].legend()

ct1 = pd.DataFrame({'label':labels,
                    'gases': y['gas'].values})
ct1 = pd.crosstab(ct1['gases'],ct1['label'], margins=True)
print(ct1)

sns.heatmap(ct1,ax=ax[1],annot=True,fmt='d',cmap='Blues')
ax[1].set_xlabel('Cluster K-Means')
ax[1].set_ylabel('Gás Real')
ax[1].set_title('Standard Scaler')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_mm)
    inertia.append(kmeans.inertia_)

ax[0].plot(range(1, 11), inertia, marker='o')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=3, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {3}")

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_mm)
    score = silhouette_score(X_mm, kmeans.labels_)
    silhouette_scores.append(score)

ax[1].plot(range(2, 11), silhouette_scores, marker='o')
ax[1].set_title('Índice de Silhueta para Diferentes Valores de K')
ax[1].set_xlabel('Número de Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].axvline(x=2, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"melhor k = {2}")
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_mm)
labels = kmeans.predict(X_mm)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    ax[0].scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
ax[0].set_xlabel('feature_9')
ax[0].set_ylabel('feature_100')
ax[0].set_title('MinMax Scaler')
ax[0].legend()

ct2 = pd.DataFrame({'label':labels,
                    'gases': y['gas'].values})
ct2 = pd.crosstab(ct2['gases'],ct2['label'], margins=True)
print(ct2)

sns.heatmap(ct2,ax=ax[1],annot=True,fmt='d',cmap='Blues')
ax[1].set_xlabel('Cluster K-Means')
ax[1].set_ylabel('Gás Real')
ax[1].set_title('MinMax Scaler')
plt.show()