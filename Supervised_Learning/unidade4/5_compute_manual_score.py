import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions,y):

    resi = y - predictions
    RSS = np.sum(resi ** 2)
    return RSS

def compute_MSE(predictions,y):

    resi = y - predictions
    MSE = np.mean(resi ** 2)
    return MSE

def compute_RMSE(predictions,y):

    resi = y - predictions
    RMSE = np.sqrt(np.mean(resi ** 2))
    return RMSE

def compute_R_squared(predictions,y):

    RSS = np.sum((y - predictions) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (RSS / TSS)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))