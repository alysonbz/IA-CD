# Perform the necessary imports
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    x_mean = x.mean()
    y_mean = y.mean()

    num = ((x - x_mean) * (y - y_mean)).sum()

    den = (((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum()) ** 0.5

    return num / den


grains_df = load_grains_dataset()

# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0]
# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1]

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
