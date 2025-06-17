import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
path = kagglehub.dataset_download("fedesoriano/stellar-classification-dataset-sdss17")
df = pd.read_csv(os.path.join(path, 'star_classification.csv'))

# Remover valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())
df = df.dropna()

# Analisar distribuição da variável-alvo
print("\nDistribuição da variável-alvo 'class':\n", df['class'].value_counts())
sns.countplot(data=df, x='class', order=df['class'].value_counts().index)
plt.title("Distribuição das Classes de Estrelas")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Codificar variáveis categóricas
categoricas = df.select_dtypes(include='object').columns
for col in categoricas:
    df[col] = LabelEncoder().fit_transform(df[col])
print("\nVariáveis categóricas codificadas:", list(categoricas))

# Estatísticas descritivas e correlação
print("\nEstatísticas descritivas:")
print(df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# Salvar dataset ajustado
df.to_csv("classificacao_ajustado.csv", index=False)
print("Dataset salvo como 'classificacao_ajustado.csv'.")
