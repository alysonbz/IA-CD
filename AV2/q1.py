import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Carregamento dos dados
df = pd.read_excel("./Av2/OnlineRetail.xlsx")

# Limpeza inicial
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Agrupamento por cliente
cliente_df = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'InvoiceDate': ['min', 'max'],
    'Country': 'first'
})
cliente_df.columns = ['NumCompras', 'TotalItens', 'PrecoMedio', 'PrimeiraCompra', 'UltimaCompra', 'Pais']
cliente_df['Recencia'] = (df['InvoiceDate'].max() - cliente_df['UltimaCompra']).dt.days
cliente_df.reset_index(inplace=True)

# Estatísticas descritivas
estatisticas = cliente_df[['NumCompras', 'TotalItens', 'PrecoMedio', 'Recencia']].describe()
print("=== Estatísticas Descritivas ===")
print(estatisticas)

# Histograma
cliente_df[['NumCompras', 'TotalItens', 'PrecoMedio', 'Recencia']].hist(bins=30, figsize=(10, 8))
plt.tight_layout()
plt.savefig("histograma_variaveis.png")

# Correlação
corr = cliente_df[['NumCompras', 'TotalItens', 'PrecoMedio', 'Recencia']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.savefig("matriz_correlacao.png")

# Justificativa: variáveis com alta correlação podem ser redundantes
# Padronização será usada pois as variáveis têm escalas diferentes
