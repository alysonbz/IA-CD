import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

# Carregar dados pré-processados
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\regressao_ajustado.csv")
X = df.drop('charges', axis=1)
y = df['charges']

# Modelos
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

# Avaliação com Cross-Validation (5 folds)
results = []
for name, model in models.items():
    neg_mse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    r2 = cross_val_score(model, X, y, scoring='r2', cv=5)
    
    rmse_scores = np.sqrt(-neg_mse)
    results.append({
        'Modelo': name,
        'RMSE Médio': rmse_scores.mean(),
        'R² Médio': r2.mean()
    })

# Exibir resultados
results_df = pd.DataFrame(results)
print("\nComparação dos Modelos:")
print(results_df.sort_values(by='RMSE Médio'))
