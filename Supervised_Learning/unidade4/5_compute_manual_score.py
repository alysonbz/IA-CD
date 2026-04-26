import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions,y):
    RSS = 0
    for i in range (len(predictions)):
        RSS += (y[i] - predictions[i])**2
    return RSS
def compute_MSE(predictions,y):
    MSE= 0
    for i in range (len(predictions)):
        MSE += (y[i] - predictions[i])**2
    MSE = MSE/len(predictions)
    return MSE
def compute_RMSE(predictions,y):
    RMSE = 0
    for i in range (len(predictions)):
        RMSE += (y[i] - predictions[i])**2
    RMSE = np.sqrt(RMSE/len(predictions))
    return RMSE
def compute_R_squared(predictions,y):
    top = 0
    bottom = 0
    for i in range (len(predictions)):
        top += (predictions[i] - np.mean(y))**2
        bottom += (y[i] - np.mean(y))**2
    r_squared = top/bottom
    return r_squared



X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))