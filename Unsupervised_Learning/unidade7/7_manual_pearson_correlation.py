# Perform the necessary imports
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    x_m = 0
    for i in range(len(x)):
        x_m += x[i]
    x_m = x_m/len(x)
    y_m = 0
    for i in range(len(y)):
        y_m += y[i]
    y_m = y_m / len(y)

    top = 0
    botton1 = 0
    botton2 = 0
    for i in range(len(x)):
        top += (x[i]-x_m)*(y[i]-y_m)
        botton1 += (x[i]-x_m)**2
        botton2 += (y[i]-y_m)**2
    botton = (botton1*botton2)**(1/2)
    r = top/botton

    return r


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
