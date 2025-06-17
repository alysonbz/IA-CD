import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset
path = kagglehub.dataset_download("quantbruce/real-estate-price-prediction")
df = pd.read_csv(os.path.join(path, 'Real estate.csv'))

# 2. Tratar valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())
df = df.dropna()

# 3. Verificar correlação com a variável-alvo
if 'Y house price of unit area' in df.columns:
    correlation = df.corr(numeric_only=True)
    target_corr = correlation['Y house price of unit area'].sort_values(ascending=False)
    print("\nCorrelação com a variável-alvo:\n", target_corr)

    # Visualizar a matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.show()
else:
    print("Coluna-alvo não encontrada. Verifique o nome da coluna.")

# 4. Normalizar (padronizar) os dados numéricos
numeric_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Salvar como 'regressao_ajustado.csv'
output_path = "regressao_ajustado.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Arquivo salvo como '{output_path}' com dados padronizados.")
