import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregando dataset (sem mexer nas colunas ainda)
df = pd.read_csv("C:/Users/jfm12/Downloads/nivaldo/IA-CD/AV1/dataset/CarPrice_Assignment.csv")
print(df.head(), '\n')
print(df.shape)

# 2. Verificando valores ausentes e removendo, se houver
print("Valores ausentes por coluna:\n", df.isnull().sum())
df = df.dropna()

# 3. Tratamento das variáveis categóricas — fatorizar colunas categóricas específicas
categorical_cols = [
    'CarName', 'fueltype', 'aspiration', 'doornumber',
    'carbody', 'drivewheel', 'enginelocation',
    'enginetype', 'cylindernumber', 'fuelsystem'
]

for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# 4. Correlação com a variável-alvo 'price'
correlation = df.corr()['price'].sort_values(ascending=False)
print("Correlação das variáveis com Price:\n", correlation)

# 5. Padronização dos dados numéricos (exceto price)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('price')

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df.head())

# 6. Salvar arquivo ajustado (opcional)
df.to_csv('regressao_ajustado.csv', index=False)
print("Arquivo 'regressao_ajustado.csv' salvo com sucesso!")