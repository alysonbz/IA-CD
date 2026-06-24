from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = load_points()

model = KMeans(n_clusters=3)

test_points = points[:50,:]
train_points = points[50:,:]

model.fit(train_points)

labels = model.predict(test_points)

print(labels)

xs = test_points[:,0]
ys = test_points[:,1]

plt.scatter(xs, ys, c=labels)

centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, s=50, marker='D')
plt.show()