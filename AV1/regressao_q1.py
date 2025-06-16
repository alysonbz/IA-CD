import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Carrega o dataset
df = pd.read_csv("fish.csv")

# Verifica valores ausentes
print(df.isnull().sum())

# One-hot encoding na coluna 'Species'
df = pd.get_dummies(df, columns=['Species'], drop_first=True)

# Calcula a correlação com o peso
correlacoes = df.corr()['Weight'].sort_values(ascending=False)
print("Correlação com o peso:\n", correlacoes)

# Matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("correlacao_matriz.png")
plt.close()

# Normalização com StandardScaler
scaler = StandardScaler()
colunas_numericas = df.drop(columns=['Weight']).columns
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# Salva dataset ajustado
df.to_csv("regressao_ajustado.csv", index=False)
