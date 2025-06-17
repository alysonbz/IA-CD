import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset original
df = pd.read_csv("/Users/luizanogueira/PycharmProjects/IA-CD/AV1/mat.csv")
df = pd.read_csv("/Users/luizanogueira/PycharmProjects/IA-CD/AV1/por.csv")

# 2. Verificar valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())

# 3. Selecionar apenas colunas numéricas
df_numeric = df.select_dtypes(include="number")

# 4. Calcular correlação com G3
correlation_matrix = df_numeric.corr()
print("\nCorrelação com G3:\n", correlation_matrix["G3"].sort_values(ascending=False))

# 5. Padronizar os dados numéricos
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# 6. Salvar dataset padronizado
df_scaled.to_csv("regressao_ajustado.csv", index=False)
print("\nArquivo 'regressao_ajustado.csv' salvo com sucesso.")

# 7. Gerar gráfico de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação entre Atributos Numéricos (Questão 1)")
plt.tight_layout()
plt.savefig("correlacao_q1_grafico.png")
plt.show()
print("Gráfico salvo como 'correlacao_q1_grafico.png'.")