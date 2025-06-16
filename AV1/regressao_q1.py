import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\insurance.csv")

# Tratar valores ausentes (verificação)
if df.isnull().sum().any():
    df = df.dropna()

# Codificar variáveis categóricas
df_codificado = pd.get_dummies(df, drop_first=True)

# Verificar correlação com a variável-alvo
correlacao = df_codificado.corr()['charges'].sort_values(ascending=False)
print("\nCorrelação com 'charges':\n", correlacao)

# Normalizar (padronizar) os dados (exceto variável-alvo)
features = df_codificado.drop('charges', axis=1)
alvo = df_codificado['charges']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Combinar novamente em um DataFrame
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
df_final = pd.concat([features_scaled_df, alvo], axis=1)

# Salvar em CSV
df_final.to_csv(r'C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\regressao_ajustado.csv', index=False)
print("\nArquivo 'regressao_ajustado.csv' salvo com sucesso.")
