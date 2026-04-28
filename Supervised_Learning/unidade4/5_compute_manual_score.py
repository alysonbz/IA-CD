import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions,y):
    sub_squared = np.square(y-predictions)
    RSS = np.sum(sub_squared)
    return RSS

def compute_MSE(predictions,y):
    RSS  = compute_RSS(predictions, y)
    MSE = np.divide(RSS, len(predictions))
    return MSE

def compute_RMSE(predictions,y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(predictions, y):
    rss = np.sum((y - predictions)**2)
    tss = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (rss / tss)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))