import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregando dataset
df = pd.read_csv("C:/Users/jfm12/Downloads/nivaldo/IA-CD/AV1/dataset/CarPrice_Assignment.csv")
print(df.head(), '\n')
print(df.shape)

# 2. Verificando valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())

# 3. Verificando a correlação com a variável-alvo (Price)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation = numeric_df.corr()['price'].sort_values(ascending=False)
print("Correlação das variáveis com Price:\n", correlation)

# Codificando variáveis categóricas (one-hot encoding)
df.drop(['CarName'], axis=1, inplace=True)

cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Normalize ou padronize os dados (excluindo a variável-alvo e variáveis categóricas)
# considerar somente as colunas numéricas para padronização

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('price')

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df.head())

# 5. Salve como regressao_ajustado.csv
df.to_csv('regressao_ajustado.csv', index=False)
print("Arquivo 'regressao_ajustado.csv' salvo com sucesso!")
