
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregamento dos dados
df = pd.read_excel('./AV2/dados/dados.xlsx')
df = df.dropna(subset=["CustomerID"])  # Remover clientes nulos
df = df[df["Quantity"] > 0]            # Remover devoluções
df = df[df["UnitPrice"] > 0]           # Remover preços negativos ou zero

# 2. Feature Engineering - Gerar variáveis agregadas por cliente
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
agg = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",          # Número de compras
    "Quantity": "sum",               # Quantidade total comprada
    "TotalPrice": "sum",             # Valor total comprado
    "InvoiceDate": ["min", "max"],   # Para calcular recência depois
}).reset_index()

agg.columns = ["CustomerID", "NumCompras", "TotalQuantidade", "TotalGasto", "PrimeiraCompra", "UltimaCompra"]
agg["RecenciaDias"] = (agg["UltimaCompra"].max() - agg["UltimaCompra"]).dt.days
agg = agg.drop(columns=["PrimeiraCompra", "UltimaCompra"])

# 3. Pré-processamento
features = ["NumCompras", "TotalQuantidade", "TotalGasto", "RecenciaDias"]
X = agg[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Clusterização - KMeans com k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
agg["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Análise de centroides
centroides = pd.DataFrame(kmeans.cluster_centers_, columns=features)
centroides = scaler.inverse_transform(centroides)
centroides_df = pd.DataFrame(centroides, columns=features)
print("\nCentroides dos clusters:")
print(centroides_df)

# 6. Boxplots comparativos por cluster
for col in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=agg, x="Cluster", y=col)
    plt.title(f"{col} por Cluster")
    plt.tight_layout()
    plt.savefig(f"{col}_boxplot.png")

# 7. PCA para visualização
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
agg["PCA1"] = components[:, 0]
agg["PCA2"] = components[:, 1]

plt.figure(figsize=(6, 5))
sns.scatterplot(data=agg, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
plt.title("Clusters com PCA")
plt.tight_layout()
plt.savefig("clusters_pca.png")

# 8. Interpretação semântica
print("\nMédia das variáveis por cluster:")
print(agg.groupby("Cluster")[features].mean())

# 9. Aplicar crosstab (Exemplo com faixa de TotalGasto)
agg["FaixaGasto"] = pd.cut(agg["TotalGasto"], bins=[0, 500, 2000, np.inf],
                           labels=["Baixo", "Médio", "Alto"])

ct = pd.crosstab(agg["Cluster"], agg["FaixaGasto"], normalize='index')
print("\nDistribuição de gasto por cluster:")
print(ct)
