import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
file_path = 'AB_NYC_2019.csv'
df = pd.read_csv(file_path)

# Remover colunas não numéricas irrelevantes para modelagem
df_clean = df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])

# Tratar valores ausentes
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

# Codificar variáveis categóricas com one-hot encoding
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Analisar correlação com a variável alvo 'price'
correlation = df_encoded.corr()['price'].sort_values(ascending=False)
print("Correlação com 'price':\n", correlation)

# Separar features (X) e alvo (y)
X = df_encoded.drop(columns='price')
y = df_encoded['price']

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converter X escalado de volta para dataframe
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Concatenar novamente com a variável alvo
final_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# Salvar o dataframe ajustado
output_path = 'regressao_ajustado.csv'
final_df.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")
