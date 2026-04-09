from src.utils import load_churn_dataset
import numpy as np

# 1.Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

churn_df = load_churn_dataset()

# 2. Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["total_day_charge", "total_eve_charge"]].values

# 3. Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# 4. Fit the classifier to the data
knn.fit(X, y)

X_test = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# 5. Predict the labels for the X_teste
y_pred = knn.predict(X_test)

# 6. Print the predictions for X_test
print("\nQuestão 6.")
print("Predictions: {}".format(y_pred))