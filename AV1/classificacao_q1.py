import pandas as pd
import matplotlib.pyplot as plt
# Carregando o dataset
df = pd.read_csv("C:/Users/jfm12/Downloads/nivaldo/IA-CD/AV1/dataset/chocolate.csv")
print(df.head(), '\n')

# tratando valores ausentes
print(df.isnull().sum(), '\n')

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna("desconhecido")
df = df.drop(columns=['Bean\nType', 'Broad Bean\nOrigin'])
print(df.isnull().sum())

# Análisando a distribuição da variável-alvo
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
# Mostrar as primeiras linhas para conferir
print(df_encoded.head())


# 5. Análise estatística exploratória
print("\nEstatísticas descritivas:")
print(df_encoded.describe())

# 6. Salvar o arquivo ajustado




