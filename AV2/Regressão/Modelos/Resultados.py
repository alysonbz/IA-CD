from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold as k

from Padrao import *
print("Regressão Padrão")
print("MAE Modelo padrão:", mean_absolute_error(ytest, y_pred))
print("MSE Modelo padrão:", mean_squared_error(ytest, y_pred))
print("RMSE Modelo padrão:", np.sqrt(mean_squared_error(ytest, y_pred)))
print("R² Modelo padrão: ", regressor.score(Xtest, ytest))
print("Validação Cruzada R² Modelo padrão:\n",cross_val_score(regressor, X, y, cv=kf))
print("Média da validação Cruzada R² Modelo padrão:", np.mean(cross_val_score(regressor, X, y, cv=kf)))
print("##################################################################")

from Ridge import *
print("\nRegressão Ridge")
print("MAE Modelo ridge:", mean_absolute_error(ytest, y_pred_r))
print("MSE Modelo ridge:", mean_squared_error(ytest, y_pred_r))
print("RMSE Modelo ridge:", np.sqrt(mean_squared_error(ytest, y_pred_r)))
print("R² Modelo ridge: ", melhor_modelo_r.score(Xtest, ytest))
print("Validação Cruzada R² Modelo ridge:\n",cross_val_score(melhor_modelo_r, X, y, cv=kf))
print("Média da validação Cruzada R² Modelo ridge:\n", ridge_cv.best_score_)
print("##################################################################")

from Lasso import *
print("\nRegressão Lasso")
print("MAE Modelo lasso:", mean_absolute_error(ytest, y_pred_l))
print("MSE Modelo lasso:", mean_squared_error(ytest, y_pred_l))
print("RMSE Modelo lasso:", np.sqrt(mean_squared_error(ytest, y_pred_l)))
print("R² Modelo lasso: ", melhor_modelo_l.score(Xtest, ytest))
print("Validação Cruzada R² Modelo lasso:\n",cross_val_score(melhor_modelo_l, X, y, cv=kf))
print("Média da validação Cruzada R² Modelo lasso:\n", lasso_cv.best_score_)
print("##################################################################")