# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    return None


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()


# Calculate the Pearson correlation
def pearson(x, y):
    media_x = np.mean(x)
    media_y = np.mean(y)
    nume = np.sum((x-media_x)*(y-media_y))
    deno_x = np.sum((x-media_x)**2)
    deno_y = np.sum((y-media_y)**2)
    deno = np.sqrt(deno_x*deno_y)

    return nume/deno

correlation = pearson(width,length)

# Display the correlation
print(correlation)
