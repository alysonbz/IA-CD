import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o dataset
df = pd.read_csv('customer_data.csv')

# 2. Tratar valores ausentes
missing = df.isnull().sum()
print("Valores ausentes por coluna:\n", missing)

# Preencher valores ausentes
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Analisar a distribuição da variável-alvo
if 'label' in df.columns:
    print("\nDistribuição da variável-alvo:")
    print(df['label'].value_counts(normalize=True))
    sns.countplot(x='label', data=df)
    plt.title('Distribuição da variável-alvo')
    plt.show()

# 4. Codificar variáveis categóricas
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. Análise estatística exploratória
print("\nEstatística:")
with pd.option_context('display.float_format', '{:.2f}'.format):
    print(df_encoded.describe())

# Correlação entre variáveis
plt.figure(figsize=(12, 8))
corr_matrix = df_encoded.corr().round(2)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# 6. Salvar dataset ajustado
df_encoded.to_csv('classificacao_ajustado.csv', index=False)
print("\nArquivo 'classificacao_ajustado.csv' salvo com sucesso.")
