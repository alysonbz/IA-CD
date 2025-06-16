import pandas as pd

#1. Carregue o dataset com pandas.
df = pd.read_csv('dataset/Cancer_Data.csv')
print(f'5 primeiras linhas do dataset:\n{df.head()}')
print(f'\nTamanho do dataset: \n{df.shape}')

#2. Trate valores ausentes.
print(f'\nValores Nulos: \n{df.isna().sum()}')
df = df.drop('Unnamed: 32', axis=1)

#3. Analise a distribuição da variável-alvo.
print('\n', df['diagnosis'].value_counts())

#4. Codifique variáveis categóricas, se necessário.
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
print('\n', df['diagnosis'].value_counts())

#5. Faça uma análise estatística exploratória
print('\nResumo estatístico do dataset:\n', df.describe())

#6. Salve o arquivo como classificacao_ajustado.csv.
df.to_csv('dataset/classificacao_ajustado.csv', index=False)
