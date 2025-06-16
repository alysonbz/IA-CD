# 1. Aplique as regressões: Linear, Ridge e Lasso.

from sklearn.linear_model import LinearRegression, Ridge, Lasso

#Linear
linear_model = LinearRegression()
linear_model.fit(X, y)


print("B1:", linear_model.coef_)
print("B0:", linear_model.intercept_)
print("Treinamento (R2):", linear_model.score(X, y))

#Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

print("B1 Ridge:", ridge_model.coef_)
print("B0 Ridge:", ridge_model.intercept_)
print("Treinamento Ridge (R2):", ridge_model.score(X, y))

#Lasso
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)

print("B1 Lasso:", lasso_model.coef_)
print("B0 Lasso:", lasso_model.intercept_)
print("Treinamento Lasso (R2):", lasso_model.score(X, y))

##

# 2. Use cross_val_score com 5 folds.

from sklearn.model_selection import cross_val_score

modelo_linear = LinearRegression()
modelo_ridge = Ridge(alpha=1.0)
modelo_lasso = Lasso(alpha=1.0)

cross_val = 5

scores_linear = cross_val_score(modelo_linear, X, y, cv=cross_val)
print("Regressão Linear R2 scores: ", scores_linear)


scores_ridge = cross_val_score(modelo_ridge, X, y, cv=cross_val)
print("Regressão Ridge R2 scores: ", scores_ridge)

scores_lasso = cross_val_score(modelo_lasso, X, y, cv=cross_val)
print("Regressão Lasso R2 scores: ", scores_lasso)

##

# 3. Compare RMSE e R2 em uma tabela.
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


modelos = { 
    "Linear": linear_model,
    "Ridge": ridge_model,
    "Lasso": lasso_model
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
resultados = []

for xp, modelos in modelos.items():
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, ytest = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    resultados.append({
        "Modelo": xp,
        "R2 Médio": np.mean(r2_scores),
        "RMSE Médio": np.mean(rmse_scores)
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados)

##

# 4. Discuta os resultados
#valores iguaisss. 