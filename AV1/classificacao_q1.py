import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

# Tratar valores ausentes
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Analisar a distribuição da variável-alvo
print("Distribuição da variável-alvo:")
print(df['class'].value_counts())

# Codificar variáveis categóricas
df_encoded = pd.get_dummies(df)

# Análise estatística exploratória
desc = df.describe()
print(desc)

# Salvar dataset ajustado
df_encoded.to_csv('classificacao_ajustado.csv', index=False)
