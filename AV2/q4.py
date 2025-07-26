#Interpretação Semântica dos Agrupamentos

#Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importando o CSV ajustado
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV2/mall_ajustado.csv')


#Exibindo as médias das variáveis numéricas por cluster
print("Médias por cluster:")
print(df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
print("________________________________________________________")

#Criando boxplots comparativos
variaveis = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for var in variaveis:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='cluster', y=var, data=df)
    plt.title(f'{var} por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(var)
    plt.show()

#Crosstab de cluster por gênero (opcional)
if 'Gender' in df.columns:
    print("Crosstab por gênero:")
    print(pd.crosstab(df['cluster'], df['Gender']))
    print("________________________________________________________")

# Conclusão:

# A análise das médias e dos boxplots por cluster permitiu interpretar semanticamente os agrupamentos.
# Cada cluster apresenta características distintas em relação à idade, renda anual e pontuação de gastos.
# Essa diferenciação possibilita nomear os grupos com base em seus perfis


