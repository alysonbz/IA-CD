import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados com clusters
df = pd.read_csv('dataset/marketing_campaign_with_clusters.csv')

# Análise por cluster (KMeans como base)
grouped = df.groupby('KMeans_Cluster').mean(numeric_only=True)

print("Média dos atributos por cluster:")
print(grouped)

# Boxplots para principais variáveis
cols = ['Income', 'Age', 'Spent', 'Customer_For', 'Family_Size']
for col in cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='KMeans_Cluster', y=col, data=df, hue='KMeans_Cluster', palette='Set2', legend=False)
    plt.title(f'Distribuição de {col} por Cluster (KMeans)')
    plt.show()

# Crosstab com Is_Parent e Education
ct1 = pd.crosstab(df['KMeans_Cluster'], df['Is_Parent'], normalize='index')
ct2 = pd.crosstab(df['KMeans_Cluster'], df['Education'], normalize='index')

print("\nDistribuição de Is_Parent por Cluster:")
print(ct1)

print("\nDistribuição de Education por Cluster:")
print(ct2)
