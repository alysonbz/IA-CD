import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Carregar dados
df = pd.read_csv('dataset/marketing_campaign_preprocessed.csv')
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Elbow e Silhouette
inertia = []
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_df)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(scaled_df, km.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title("Elbow - Inertia")
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title("Silhouette Score")
plt.tight_layout()
plt.show()

# KMeans com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_df)

# Hierarchical
linked_avg = linkage(scaled_df, method='average')
linked_complete = linkage(scaled_df, method='complete')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
dendrogram(linked_avg, truncate_mode='lastp', p=10)
plt.title("Dendrograma Average")
plt.subplot(1, 2, 2)
dendrogram(linked_complete, truncate_mode='lastp', p=10)
plt.title("Dendrograma Complete")
plt.tight_layout()
plt.show()

# Clusterização hierárquica
agg_avg = AgglomerativeClustering(n_clusters=4, linkage='average')
agg_complete = AgglomerativeClustering(n_clusters=4, linkage='complete')
df['Hierarchical_Avg'] = agg_avg.fit_predict(scaled_df)
df['Hierarchical_Complete'] = agg_complete.fit_predict(scaled_df)

df.to_csv('dataset/marketing_campaign_with_clusters.csv', index=False)
print("Clusterização concluída e salva.")
