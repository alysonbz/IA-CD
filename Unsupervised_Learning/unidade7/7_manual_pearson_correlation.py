# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    val_x = x - np.mean(x)
    val_y = y - np.mean(y)
    raiz = np.sqrt((val_x**2) * val_y**2)
    return sum(val_x * val_y) / raiz


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
