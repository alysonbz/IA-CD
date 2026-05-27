from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold as k
from sklearn.model_selection import GridSearchCV
import numpy as np
from Dataset.Pre_processamento import *

Xtrain, Xtest, ytrain, ytest = train_test_split(X_norm, y, test_size=0.2, random_state=42)

kf = k(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.linspace(0.001, 1, 100)}
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)
lasso_cv.fit(Xtrain, ytrain)
melhor_modelo_l = lasso_cv.best_estimator_
y_pred_l = melhor_modelo_l.predict(Xtest)
#print(lasso_cv.best_params_)#0.01109
#print(lasso_cv.best_score_)