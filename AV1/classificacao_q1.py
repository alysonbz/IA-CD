import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Carregar o dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/mushroom.csv')

# 2. Verificar valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# 3. Análise da variável-alvo
print("\nDistribuição da classe:")
print(df['class'].value_counts())

sns.countplot(x='class', data=df)
plt.title("Distribuição das classes (comestível vs venenoso)")
plt.savefig("distribuicao_classe.png")
plt.close()

# 4. Codificar variáveis categóricas
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# 5. Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df_encoded.describe())

# 6. Mapa de calor das correlações
sns.heatmap(df_encoded.corr(), cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.savefig("correlacao_matriz.png")
plt.close()

# 7. Salvar o dataset ajustado
df_encoded.to_csv("classificacao_ajustado.csv", index=False)
print("\nArquivo 'classificacao_ajustado.csv' salvo com sucesso.")
