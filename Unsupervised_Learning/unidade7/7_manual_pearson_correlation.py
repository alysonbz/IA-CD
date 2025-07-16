# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
        soma_xy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
        soma_x2 = np.sum((x - np.mean(x)) ** 2)
        soma_y2 = np.sum((y - np.mean(y)) ** 2)
        r = soma_xy / np.sqrt(soma_x2 * soma_y2)
        return r

grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df.iloc[:,0]

# Assign the 1st column of grains: length
length = grains_df.iloc[:,1]


# Calculate the Pearson correlation
correlation = pearson_correlation(width, length)

# Display the correlation
print(correlation)
