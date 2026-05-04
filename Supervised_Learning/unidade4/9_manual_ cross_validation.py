import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X, y):
        # Treina uma das frações
        obj.fit(X, y)
        return obj.score(X, y)

    def cross_val_score(self, obj, X, y):
        scores = []

        # Parte 1: Dividir o dataset X em n_splits partes

        X_folds = np.array_split(X, self.n_splits)
        y_folds = np.array_split(y, self.n_splits)

        # Parte 2: Calcular a métrica para cada subset
        for i in range(self.n_splits):

            X_subset = X_folds[i]
            y_subset = y_folds[i]




            score = self._compute_score(obj, X_subset, y_subset)


            scores.append(score)

        # Parte 3: Retornar a lista de scores
        return scores


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg, X, y)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))