import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold

# Carregar o dataset já ajustado
df = pd.read_csv('regressao_ajustado.csv')

# Separar as variáveis independentes (X) e a dependente (y)
X = df.drop(columns='price')
y = df['price']

# Definir os modelos de regressão
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

# Configurar validação cruzada com 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lista para armazenar os resultados
results = []

# Avaliar cada modelo
for name, model in models.items():
    # Calcular o R²
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    # Calcular o RMSE (o sklearn retorna o negativo, por isso usamos -)
    rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')

    # Armazenar os resultados médios
    results.append({
        'Modelo': name,
        'R2 Médio': round(np.mean(r2_scores), 4),
        'RMSE Médio': round(np.mean(rmse_scores), 2)
    })

# Criar um DataFrame com os resultados
results_df = pd.DataFrame(results)

# Exibir a tabela
print(results_df)

# ---------------------- COMENTÁRIO SOBRE OS RESULTADOS ----------------------
# Comentários:
# 
# Em questão, todos os modelos apresentaram resultados praticamente idênticos,
# tanto no R² (aproximadamente 9,8%) quanto no RMSE (aproximadamente 226).
# 
# Isso indica que a regularização aplicada pelo Ridge e Lasso não trouxe
# melhorias significativas, sugerindo que o problema não está no overfitting,
# mas sim na limitação das variáveis disponíveis para explicar o preço.
# 
# Na escolha do melhor modelo, como todos tiveram desempenho praticamente igual,
# o mais adequado é escolher a Regressão Linear Simples, pois ela é mais
# interpretável, direta e menos complexa, já que as regularizações não
# trouxeram ganhos neste caso.
# -----------------------------------------------------------------------------
