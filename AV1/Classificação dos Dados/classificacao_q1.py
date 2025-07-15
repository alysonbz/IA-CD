import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar o dataset
file_path = 'payment_data.csv'
df = pd.read_csv(file_path)

# Exibir as primeiras linhas do dataset
print(df.head(10))

# 2. Verificar valores ausentes por coluna
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 3. Tratamento de valores ausentes
print("\nValores ausentes antes do tratamento:")
print(df.isnull().sum())

df['prod_limit'] = df['prod_limit'].fillna(df['prod_limit'].median())
df['highest_balance'] = df['highest_balance'].fillna(df['highest_balance'].median())
df['update_date'] = df['update_date'].fillna(df['update_date'].mode()[0])
df['report_date'] = df['report_date'].fillna(df['report_date'].mode()[0])

print("\nValores ausentes após o tratamento:")
print(df.isnull().sum())

# 4. Análise da distribuição da variável-alvo
sns.countplot(x='OVD_sum', data=df)
plt.title('Distribuição da Variável-Alvo: OVD_sum')
plt.xlabel('OVD_sum')
plt.ylabel('Contagem')
plt.show()

# 5. Codificação de variáveis categóricas
df['update_date'] = pd.to_datetime(df['update_date'], dayfirst=True)
df['report_date'] = pd.to_datetime(df['report_date'], dayfirst=True)
df['prod_code'] = df['prod_code'].astype('category').cat.codes

# 6. Análise estatística exploratória
print(df.describe())
print(df.corr(numeric_only=True))

# 7. Salvando o dataset ajustado
df.to_csv('classificacao_ajustado.csv', index=False)
print("\nArquivo salvo como classificacao_ajustado.csv")
