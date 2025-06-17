import pandas as pd
import numpy as np

# Carregar dataset
df = pd.read_csv('/root/.cache/kagglehub/datasets/vipullrathod/fish-market/versions/1/Fish.csv')

# Tratar valores ausentes (se houver)
df.dropna(inplace=True)

# Remover coluna categórica antes da correlação
df_numeric = df.select_dtypes(include=[np.number])

# Verificar correlação com variável alvo 'Weight'
correlation = df_numeric.corr()['Weight'].sort_values(ascending=False)
print("Correlação das variáveis numéricas com Weight:")
print(correlation)

# Normalizar dados (excluindo variável alvo)
from sklearn.preprocessing import MinMaxScaler

X = df_numeric.drop('Weight', axis=1)
y = df_numeric['Weight']

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Juntar variável alvo
df_normalized = pd.concat([X_normalized, y.reset_index(drop=True)], axis=1)

# Salvar arquivo ajustado
df_normalized.to_csv('regressao_ajustado.csv', index=False)
