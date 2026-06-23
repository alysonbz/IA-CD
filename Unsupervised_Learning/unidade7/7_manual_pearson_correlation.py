# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2)*np.sum((y - mean_y) ** 2))

    return numerator / denominator

grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0]

# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1]

plt.scatter(length, width)
plt.xlabel('Length')
plt.ylabel('Width')
plt.show()

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
