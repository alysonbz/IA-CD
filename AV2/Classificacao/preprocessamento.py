import pandas as pd

#importando o dataset
pd.set_option('display.max_columns', None)
df = pd.read_csv('online_shoppers_intention.csv')
print(df.head())

#Tamanho do Dataset
print('Tamanho do Dataset:', df.shape)

#Informações Gerais
print(df.info())

#Descrição do Dataset
print(df.describe())

#Quantidade de Valores Nulos
print(df.isnull().sum())

#Verificação de possíveis valores negativos
numericas = df.select_dtypes(include=['int64', 'float64'])
print((numericas < 0).sum())

#Ver quantidade de valores duplicados
print('Total de valores duplicados: ', df.duplicated().sum())

#Remoção de Valores Duplicados
print('Antes: ', df.shape)
df = df.drop_duplicates()
print('Depois: ', df.shape)

#Ver valores únicos categóricos
print(df["Month"].unique())
print(df["VisitorType"].unique())
print(df["Weekend"].unique())

#Converter valores booleanos em numéricos
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

#Converter valores categóricos em numéricos
df = pd.get_dummies(df, columns=['Month', 'VisitorType'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(df.head())
print(df.info())

#Salvar o dataset após o pre-processamento
df.to_csv("dataset_tratado.csv", index=False)