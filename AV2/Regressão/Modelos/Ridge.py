from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold as k
from sklearn.model_selection import GridSearchCV
import numpy as np
from Dataset.Pre_processamento import *

Xtrain, Xtest, ytrain, ytest = train_test_split(X_norm, y, test_size=0.2, random_state=42)

kf = k(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.linspace(0.001, 10, 100)}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(Xtrain, ytrain)
melhor_modelo_r = ridge_cv.best_estimator_
y_pred_r = melhor_modelo_r.predict(Xtest)
#print(cross_val_score(best_model, X, y, cv=kf))
#print(ridge_cv.best_params_)#2.223
#print(ridge_cv.best_score_)