import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
df = pd.read_excel('./AV2/dados/dados.xlsx')

# 2. Limpeza inicial (remoção de dados ausentes relevantes)
df = df.dropna(subset=['CustomerID', 'Description'])

# 3. Criação de nova coluna de receita
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 4. Estatísticas descritivas das variáveis numéricas
numeric_cols = ['Quantity', 'UnitPrice', 'TotalPrice']
print(df[numeric_cols].describe())

# 5. Visualização das distribuições
plt.figure(figsize=(15, 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribuição de {col}')
plt.tight_layout()
plt.show()

# 6. Matriz de correlação
corr = df[numeric_cols].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlação entre variáveis numéricas")
plt.show()

# 7. Detecção de outliers simples (boxplots)
plt.figure(figsize=(15, 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()

# 8. Remoção de outliers (baseado em TotalPrice > 0 e limites razoáveis)
df = df[(df['TotalPrice'] > 0) & (df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 9. Preparação para PCA
X = df[numeric_cols]
X_scaled = StandardScaler().fit_transform(X)

# 10. PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 11. Variância explicada
explained_variance = pca.explained_variance_ratio_
print("Variância explicada por componente:", explained_variance)

# 12. Visualização do scree plot (variância explicada)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Componente Principal')
plt.ylabel('Variância Explicada')
plt.title('Scree Plot - PCA')
plt.grid(True)
plt.show()
