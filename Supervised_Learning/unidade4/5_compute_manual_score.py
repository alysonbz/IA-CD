import numpy as np
from src.utils import processing_all_features_sales_clean

#fazer manual
def compute_RSS(predictions,y):
    RSS = np.sum((y - predictions)**2)
    return RSS

def compute_MSE(predictions,y):
    MSE= np.mean((y - predictions)**2)
    return MSE

def compute_RMSE(predictions,y):
    RMSE = np.sqrt(np.mean((y - predictions)**2))
    return RMSE

def compute_R_squared(predictions,y):
    RSS = np.sum((y - predictions) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (RSS / TSS)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()

#É pra dar igual ao resultado 4_fit_prediction. A 4 é com biblioteca e a 5 é a versão manual.
print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))