import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions, y):
    y = np.array(y)
    predictions = np.array(predictions)
    residuos = predictions - y
    RSS = np.sum(residuos ** 2)
    return RSS

def compute_MSE(predictions, y):
    y = np.array(y)
    predictions = np.array(predictions)
    MSE = np.mean((predictions - y) ** 2)
    return MSE

def compute_RMSE(predictions, y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(predictions, y):
    y = np.array(y)
    predictions = np.array(predictions)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


X, y, predictions = processing_all_features_sales_clean()

print("RSS: {:.4f}".format(compute_RSS(predictions, y)))
print("MSE: {:.4f}".format(compute_MSE(predictions, y)))
print("RMSE: {:.4f}".format(compute_RMSE(predictions, y)))
print("R^2: {:.4f}".format(compute_R_squared(predictions, y)))
