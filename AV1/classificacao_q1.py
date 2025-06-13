import pandas as pd
import matplotlib.pyplot as plt

# 1.Carregando o dataset
df = pd.read_csv("C:/Users/jfm12/Downloads/nivaldo/IA-CD/AV1/dataset/chocolate.csv")
print(df.head(), '\n')

# 2.tratando valores ausentes
print(df.isnull().sum(), '\n')
# removendo colunas com valores alsentes
df.dropna(axis=1, inplace=True)
print(df.isnull().sum())

# 3.Análisando a distribuição da variável-alvo
plt.hist(df['Rating'], bins=20, color='darkBlue', edgecolor='black')
plt.title('Distribuição das Notas (Rating)')
plt.xlabel('Rating')
plt.ylabel('Frequência')
plt.show()

# 4.avaliar variaveis categóricas
categorical_cols = ['Company \n(Maker-if known)', 'Specific Bean Origin\nor Bar Name',
                    'Cocoa\nPercent', 'Company\nLocation']
# Aplicar codificação one-hot para as variáveis categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df_encoded.head())

# 5. Análise estatística exploratória
print("\nEstatísticas descritivas:")
print(df_encoded.describe())

# 6. Salvar o arquivo ajustado
df_encoded.to_csv("classificacao_ajustado.csv", index=False)


