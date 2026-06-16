# Perform the necessary imports
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset
import numpy as np


def pearson_correlation(x,y):
    x = np.array(x)
    y = np.array(y)

    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)

    return np.sum(x_diff * y_diff) / np.sqrt(np.sum(x_diff ** 2) * np.sum(y_diff ** 2))



grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0]

# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1]

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
