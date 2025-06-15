import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregando o dataset
df = pd.read_csv("C:/Users/jfm12/Downloads/nivaldo/IA-CD/AV1/dataset/chocolate.csv")
print(df.head(), '\n')

# ajeitando erros de formatação
df.columns = df.columns.str.strip().str.replace('\n', '_', regex=True)
df.columns = df.columns.str.strip().str.replace(' ', '_', regex=True)

# 2. Tratando valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum(), '\n')
print("\nNomes das colunas após limpeza:")
print(df.columns)

# Removendo colunas com valores ausentes
df.dropna(axis=1, inplace=True)
print("\nValores ausentes após remoção de colunas:")
print(df.isnull().sum())

# 3. Analisando a distribuição da variável-alvo
plt.hist(df['Rating'], bins=20, color='darkblue', edgecolor='black')
plt.title('Distribuição das Notas (Rating)')
plt.xlabel('Rating')
plt.ylabel('Frequência')
plt.show()

# 4. Codificando as variaveis categoricas.
# Limpando a coluna de percentual
df['Cocoa_Percent'] = df['Cocoa_Percent'].apply(lambda x: float(str(x).strip('%')))

# Transformando variáveis categóricas com fatoração
df['Company _(Maker-if_known)'] = pd.factorize(df['Company _(Maker-if_known)'])[0]
df['Company_Location'] = pd.factorize(df['Company_Location'])[0]
df['Specific_Bean_Origin_or_Bar_Name'] = pd.factorize(df['Specific_Bean_Origin_or_Bar_Name'])[0]

# 5. Análise estatística descritiva
print("\nEstatísticas descritivas das variáveis numéricas:")
print(df.describe())

# 6. Salvar o dataset ajustado (se quiser salvar)
df.to_csv("classificacao_ajustado.csv", index=False)
