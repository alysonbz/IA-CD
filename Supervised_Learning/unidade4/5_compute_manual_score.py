import numpy as np
import numpy as np
from src.utils import processing_all_features_sales_clean


def compute_RSS(predictions, y):
    erro = predictions - y
    RSS = np.sum(erro ** 2)
    return RSS


def compute_MSE(predictions, y):
    erro = predictions - y
    MSE = np.mean(erro ** 2)
    return MSE


def compute_RMSE(predictions, y):
    erro = predictions - y
    MSE = np.mean(erro ** 2)
    RMSE = np.sqrt(MSE)
    return RMSE


def compute_R_squared(predictions, y):
    erro = predictions - y

    RSS = np.sum(erro ** 2)

    media_y = np.mean(y)
    TSS = np.sum((y - media_y) ** 2)

    r_squared = 1 - (RSS / TSS)

    return r_squared


X, y, predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions, y)))
print("MSE: {}".format(compute_MSE(predictions, y)))
print("RMSE: {}".format(compute_RMSE(predictions, y)))
print("R^2: {}".format(compute_R_squared(predictions, y)))

#calcular o RSS de maneira manual