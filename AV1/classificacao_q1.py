# 1. Carregue o dataset com pandas.
import pandas as pd
df = pd.read_csv("/content/diabetes.csv")
df

##

# 2. Trate valores ausentes.
print("Valores ausentes: \n", df.isnull().sum()) # Verifica se há valores vazios na tabela.
print("Valores duplicados: \n", df.duplicated().sum()) # Verifica se há valores duplicados na tabela.

## 

# 3. Analise a distribuição da variável-alvo.

#Gráfico de barras com a contagem dos valores da coluna Outcome (Variável-Alvo)

import matplotlib.pyplot as plt
plt.hist(df["Outcome"])
plt.xlabel("Outcome")
plt.ylabel("Frequência")
plt.title("Distribuição da Variável Alvo")
plt.show()

##

# 4. Codifique variáveis categóricas, se necessário.
# Não tem variáveis categóricas nesse dataset

##

# 5. Faça uma análise estatística exploratória
print("Estatísticas: ", df.describe())