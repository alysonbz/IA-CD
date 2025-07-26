#Análise Exploratória com Foco em Redução de Complexidade

#Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#Importando o csv com o pandas
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV2/Mall_Customers.csv')
print("________________________________________________________")

Gender ={
    'Male' : 0,
    'Female' : 1,

}
df['Gender'] = df['Gender'].map(Gender)

#Mostrando as 5 primeiras linhas e as informações dele
print(df.head())
print("________________________________________________________")
print(df.info())
print("________________________________________________________")

#Analisando as medidas estatísticas da variáveis quantitativas
pd.set_option('display.max_columns', None)
print(df.describe(include='all'))
print("________________________________________________________")

#Contagem de quantas amostras são do sexo masculino e feminino
print(df["Gender"].value_counts())
print("\n")
print(df["Gender"].value_counts(normalize=True))
print("________________________________________________________")

#Plotando um gráfico para a distribuição de frequência da renda anual
plt.hist(df['Annual Income (k$)'], bins=4, color='blue', edgecolor='black')
plt.title('Histograma de Renda anual')
plt.xlabel('Renda anual')
plt.ylabel('Frequência')
plt.show()
print("________________________________________________________")

#Plotando um gráfico para a distribuição de frequência da pontuação de gastos
plt.hist(df['Spending Score (1-100)'], bins=4, color='blue', edgecolor='black')
plt.title('Histograma de Pontuação de gastos')
plt.xlabel('Pontuação de gastos')
plt.ylabel('Frequência')
plt.show()
print("________________________________________________________")

#Plotando um gráfico para a distribuição de frequência da idade
plt.hist(df['Age'], bins=4, color='blue', edgecolor='black')
plt.title('Histograma da Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()
print("________________________________________________________")

#Verificando valores unicos
print("Valores únicos por coluna:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} valores únicos")


#Analisando outliers com boxplot
variaveis_numericas = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in variaveis_numericas:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot de {col}')
    plt.show()

#Matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

#Padronazação com standardscaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[variaveis_numericas])

df_scaled = pd.DataFrame(df_scaled, columns=variaveis_numericas)
print(df_scaled.head())

#salvando o dataset ajustado
df_scaled.to_csv("mall_ajustado.csv", index=False)


# Conclusão da Análise Exploratória:

# - As distribuições revelaram uma boa dispersão nas variáveis 'Age', 'Annual Income' e 'Spending Score',
#   com possíveis grupos distintos de clientes.
# - A matriz de correlação mostrou baixa correlação entre as variáveis.
# - A padronização foi aplicada com sucesso, tornando as variáveis comparáveis entre si.

