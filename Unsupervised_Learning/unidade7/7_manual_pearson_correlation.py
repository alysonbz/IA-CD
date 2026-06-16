# Perform the necessary imports
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset


def pearson_correlation(width, length):
    n = len(width)

    x_mean = sum(width) / n
    y_mean = sum(length) / n

    numerator   = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(width, length))
    denominator = (
        sum((xi - x_mean) ** 2 for xi in width) *
        sum((yi - y_mean) ** 2 for yi in length)
    ) ** 0.5

    return numerator / denominator


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
