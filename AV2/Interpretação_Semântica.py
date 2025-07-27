# ===========================================
# Interpretação Semântica dos Clusters
# ===========================================

# Consideraremos K-Means com o melhor k (definido anteriormente)
df_clusters = df.copy()
df_clusters['Cluster'] = labels_kmeans  # labels_kmeans obtidos na Parte 3

# 1. Médias por cluster (somente variáveis numéricas)
cluster_means = df_clusters.groupby('Cluster')[['math score', 'reading score', 'writing score']].mean()
print("\nMédias por cluster:")
print(cluster_means)

# 2. Boxplots comparativos
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='math score', data=df_clusters)
plt.title('Distribuição de Math Score por Cluster')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='reading score', data=df_clusters)
plt.title('Distribuição de Reading Score por Cluster')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='writing score', data=df_clusters)
plt.title('Distribuição de Writing Score por Cluster')
plt.show()

# 3. Centroides (já temos do KMeans)
print("\nCentroides dos clusters (em escala padronizada):")
print(kmeans_final.cluster_centers_)

# 4. Crosstab com variáveis categóricas
# Exemplo: gênero
crosstab_gender = pd.crosstab(df_clusters['Cluster'], df_clusters['gender'])
print("\nCrosstab Cluster x Gênero:")
print(crosstab_gender)

# Crosstab com nível educacional dos pais
crosstab_parent = pd.crosstab(df_clusters['Cluster'], df_clusters['parental level of education'])
print("\nCrosstab Cluster x Education:")
print(crosstab_parent)
