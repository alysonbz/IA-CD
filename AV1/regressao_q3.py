import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# 1. Carregar o dataset ajustado
df = pd.read_csv('dataset/regressao_ajustado.csv')

# 2. Separar X e y
X = df.drop(columns=['cnt'])
y = df['cnt']

# 3. Definir modelos
modelos = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000)
}


# 4. Criar funções de avaliação
def rmse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return -np.sqrt(mean_squared_error(y, y_pred))  # Negativo pois cross_val_score maximiza


r2_scorer = make_scorer(r2_score)

# 5. Aplicar validação cruzada com 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

for nome, modelo in modelos.items():
    scores_rmse = cross_val_score(modelo, X, y, cv=kf, scoring=rmse_scorer)
    scores_r2 = cross_val_score(modelo, X, y, cv=kf, scoring=r2_scorer)

    resultados.append({
        'Modelo': nome,
        'RMSE Médio': -scores_rmse.mean(),
        'R² Médio': scores_r2.mean()
    })

# 6. Criar e exibir tabela de comparação
tabela_resultados = pd.DataFrame(resultados).sort_values(by='R² Médio', ascending=False)
print("\nComparação dos Modelos:")
print(tabela_resultados.to_string(index=False))

# 7. Identificar o melhor modelo
melhor_modelo = tabela_resultados.iloc[0]
print(
    f"\n🔍 Melhor modelo: {melhor_modelo['Modelo']} com R² médio de {melhor_modelo['R² Médio']:.4f} e RMSE médio de {melhor_modelo['RMSE Médio']:.2f}")
