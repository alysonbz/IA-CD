import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(y_true,y_pred):
    RSS = np.sum((y - predictions) ** 2)
    return RSS
def compute_MSE(y_true,y_pred):
    MSE= np.mean((y - predictions) ** 2)
    return MSE
def compute_RMSE(y_true,y_pred):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE
def compute_R_squared(y_true,y_pred):
    RSS = compute_RSS(predictions, y)
    TSS = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (RSS / TSS)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))