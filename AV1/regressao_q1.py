import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Carregue o dataset
df = pd.read_csv('dataset/day.csv')

# 2. Trate valores ausentes
# Verifica valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())

# 3. Verifique a correlação com a variável-alvo ('cnt')
correlacoes = df.corr(numeric_only=True)
cor_target = correlacoes[['cnt']].sort_values(by='cnt', ascending=False)
print("\nCorrelação com a variável-alvo 'cnt':\n", cor_target)

# Plotando o mapa de calor das correlações
plt.figure(figsize=(12, 8))
sns.heatmap(correlacoes, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.show()

# 4. Normalize ou padronize os dados
# Seleciona colunas numéricas (excluindo colunas categóricas e a target temporariamente)
colunas_excluir = ['instant', 'dteday', 'casual', 'registered']  # 'casual' e 'registered' são componentes da 'cnt'
X = df.drop(columns=colunas_excluir)
y = df['cnt']  # variável alvo

# Padronizando
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop('cnt', axis=1))

# Cria novo DataFrame padronizado com a variável alvo
df_ajustado = pd.DataFrame(X_scaled, columns=X.drop('cnt', axis=1).columns)
df_ajustado['cnt'] = y.values

# 5. Salve como regressao_ajustado.csv
df_ajustado.to_csv('dataset/regressao_ajustado.csv', index=False)