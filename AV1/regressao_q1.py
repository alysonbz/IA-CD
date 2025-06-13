import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
df = pd.read_csv('housing.csv')

# 2. Tratar valores ausentes
print("\nVerificando valores ausentes:")
print(df.isnull().sum())

# 3. Verificar correlação com a variável-alvo (MEDV)
print("\nCorrelações com MEDV:")
correlations = df.corr()[['MEDV']].sort_values(by='MEDV', ascending=False)
print(correlations)

# 4. Normalizar/padronizar os dados (exceto a variável target)
scaler = StandardScaler()
features = df.drop('MEDV', axis=1)
features_scaled = scaler.fit_transform(features)

# Criar novo DataFrame com dados padronizados
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['MEDV'] = df['MEDV'].values

# 5. Salvar dataset ajustado
df_scaled.to_csv('regressao_ajustado.csv', index=False)
print("\nDataset ajustado salvo como 'regressao_ajustado.csv'")