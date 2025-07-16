from random import sample

from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = load_points()

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

test_points = points[:50,:]
train_points = points[50:,:]

# Fit model to train_points
model.fit(train_points)

# Determine the cluster labels of new_points: labels
labels = model.predict(train_points)

# Print cluster labels of new_points
print(labels)


# Assign the columns of test_points: xs and ys
xs = test_points[:,0]
ys = test_points[:,1]

# # Make a scatter plot of xs and ys, using labels to define the colors
# plt.scatter(xs,ys,s=50, marker = 'D')

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, c=labels, s=50, marker='D')
# xs = sample[:,0]
# ys =sample[:,2]
plt.show()
