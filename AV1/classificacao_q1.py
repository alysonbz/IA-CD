#Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importando o csv com o pandas
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV1/dataset.csv')
print("________________________________________________________")

#Traduzindo as colunas para português
df.rename(columns={"Age": "Idade",
                     "Sex": "Sexo",
                     "BP": "PA",
                     "Cholesterol": "Colesterol",
                     "Na_to_K": "Razão_KA_K",
                     "Drug": "Medicamento"}, inplace=True)
print("________________________________________________________")

Medicamento = {
    'DrugY' : 0,
    'drugX' : 1,
    'drugA': 2,
    'drugC' : 3,
    'drugB' : 4,

    }

df['Medicamento'] = df['Medicamento'].map(Medicamento)

#Mostrando as 5 primeiras linhas e as informações dele
print(df.head())
print(df.info())
print("________________________________________________________")

#Verificando se tem linha com valores ausentes
print(df.isnull().sum())
print("________________________________________________________")

#Analisando as medidas estatísticas da variáveis quantitativas

print(df.describe())
print("________________________________________________________")

#Contagem de quantas amostras são do sexo masculino e feminino
print(df["Sexo"].value_counts())
print("\n")
print(df["Sexo"].value_counts(normalize=True))
print("________________________________________________________")

#Contagem de quantas amostras são high, low e normal
print(df["PA"].value_counts())
print("\n")
print(df["PA"].value_counts(normalize=True))
print("________________________________________________________")

#Contagem de quantas amostras são high e normal
print(df["Colesterol"].value_counts())
print("\n")
print(df["Colesterol"].value_counts(normalize=True))
print("________________________________________________________")

#Verificando a idade máxima e a mínima
print("Idade máxima:", df["Idade"].max())
print("Idade mínima:", df["Idade"].min())
print("________________________________________________________")

#Verificando a Razão de KA por K máxima e mínima
print("Razão_KA_K máxima:", df["Razão_KA_K"].max())
print("Razão_KA_K mínima:", df["Razão_KA_K"].min())
print("________________________________________________________")

#Plotando um gráfico para saber a distribuição de frequência dos medicamentos
plt.hist(df['Medicamento'], bins=4, color='blue', edgecolor='black')
plt.title('Histograma de medicamento')
plt.xlabel('Medicamento')
plt.ylabel('Frequência')
plt.show()
print("________________________________________________________")

#Analisando qual idade usa determinado remédio
sns.boxplot(y=df["Idade"], x=df["Medicamento"])
plt.title("Boxplot da idade x medicamento")
plt.ylabel("Idade")
plt.xlabel("Medicamento")
plt.show()
print("________________________________________________________")


sns.boxplot(y=df["Razão_KA_K"], x=df["Medicamento"])
plt.title("Boxplot da Razão_KA_K x medicamento")
plt.ylabel("Razão_KA_K")
plt.xlabel("Medicamento")
plt.show()
print("________________________________________________________")

#Analisando a distribuição de medicamento com PA
tabela = pd.crosstab(df['Medicamento'],df['PA'])
tabela.plot(kind='bar')
plt.title("Distribuição da Pressão Arterial por Medicamentos")
plt.xlabel('Medicamento')
plt.ylabel('Frequência')
plt.legend(title='Pressão Arterial(PA)')
plt.tight_layout()
plt.show()
print("________________________________________________________")

#Analisando a distribuição de medicamento com Colesterol
tabela = pd.crosstab(df['Medicamento'],df['Colesterol'])
tabela.plot(kind='bar')
plt.title("Distribuição Colesterol por Medicamentos")
plt.xlabel('Medicamento')
plt.ylabel('Frequência')
plt.legend(title='Colesterol')
plt.tight_layout()
plt.show()
print("________________________________________________________")

#Analisando a distribuição de medicamento com Sexo
tabela = pd.crosstab(df['Medicamento'],df['Sexo'])
tabela.plot(kind='bar')
plt.title("Distribuição Sexo por Medicamentos")
plt.xlabel('Medicamento')
plt.ylabel('Frequência')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()
print("________________________________________________________")

df.to_csv("classificacao_ajustado.csv", index=False)