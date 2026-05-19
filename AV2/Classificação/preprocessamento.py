import pandas as pd

# IMPORTANDO O DATASET
pd.set_option('display.max_columns', None)
df = pd.read_csv('online_shoppers_intention.csv')
print(df.head())

# TAMANHO DO DATASET
print('Tamanho do Dataset:', df.shape)

# INFORMAÇÕES GERAIS
print(df.info())

# DESCRIÇÃO DO DATASET
print(df.describe())

# QUANTIDADE DE VALORES NULOS
print(df.isnull().sum())

# VERIFICAÇÃO DE POSSÍVEIS VALORES NEGATIVOS
numericas = df.select_dtypes(include=['int64', 'float64'])
print((numericas < 0).sum())

# QUANTIDADE DE VALORES DUPLICADOS
print('Total de valores duplicados: ', df.duplicated().sum())

# REMOÇÃO DE VALORES DUPLICADOS
print('Antes: ', df.shape)
df = df.drop_duplicates()
print('Depois: ', df.shape)

# VER VALORES ÚNICOS CATEGÓRICOS
print(df["Month"].unique())
print(df["VisitorType"].unique())
print(df["Weekend"].unique())

# CONVERTER VALORES BOOLEANOS EM NUMÉRICOS
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# CONVERTER VALORES CATEGÓRICOS EM NUMÉRICOS
df = pd.get_dummies(df, columns=['Month', 'VisitorType'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(df.head())
print(df.info())

# SALVAR O DATASET APÓS O PRE-PROCESSAMENTO
df.to_csv("dataset_tratado.csv", index=False)