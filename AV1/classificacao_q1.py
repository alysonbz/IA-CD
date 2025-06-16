import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregue o dataset com pandas.
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\voice.csv")
print(df.head())

# Trate valores ausentes.
print("Valores ausentes antes do tratamento:")
print(df.isnull().sum())

df = df.dropna()


#Analise a distribuição da variável-alvo.
print(df['label'].value_counts())

#Codifique variáveis categóricas, se necessário.
print(df.dtypes)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
print(df.dtypes)


#Faça uma análise estatística exploratória
print(df.describe())

#Salve o arquivo como `classificacao_ajustado.csv`.
df.to_csv(r'C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\classificacao_ajustado.csv', index=False)