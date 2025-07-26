# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    sub_x = x-np.mean(x)
    sub_y = y-np.mean(y)
    pot_x = sub_x*sub_x
    pot_y = sub_y*sub_y
    return np.sum(sub_x*sub_y) / np.sqrt(np.sum(pot_x)*np.sum(pot_y))



grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = np.array(grains_df['0'].values)

# Assign the 1st column of grains: length
length = np.array(grains_df['1'].values)

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
