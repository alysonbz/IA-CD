from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold as k
from Dataset.Pre_processamento import *

regressor = LinearRegression()
regressor.fit(Xtrain, ytrain)
y_pred = regressor.predict(Xtest)

kf = k(n_splits=5,shuffle = True, random_state=42)