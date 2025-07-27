import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA

cliente_df['Cluster'] = labels_km

# Médias por cluster
cluster_mean = cliente_df.groupby('Cluster')[['NumCompras', 'TotalItens', 'PrecoMedio', 'Recencia']].mean()
print("Média por Cluster:\n", cluster_mean)

# Boxplot comparativo
sns.boxplot(data=cliente_df, x='Cluster', y='NumCompras')
plt.title("Distribuição de Compras por Cluster")
plt.savefig("boxplot_cluster.png")
import matplotlib.pyplot as plt

cluster_mean.plot(kind='bar', figsize=(10,6))
plt.title('Média das variáveis por cluster')
plt.xlabel('Cluster')
plt.ylabel('Média')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("media_por_cluster_bar.png")
plt.show()

