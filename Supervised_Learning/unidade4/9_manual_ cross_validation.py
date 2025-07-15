import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, y_true, y_pred):
        # Métrica de erro quadrático médio
        return np.mean((y_true - y_pred) ** 2)

    def cross_val_score(self, model, X, y):
        tamanho = len(X) // self.n_splits
        folds_X = [X[i*tamanho:(i+1)*tamanho] for i in range(self.n_splits)]
        folds_y = [y[i*tamanho:(i+1)*tamanho] for i in range(self.n_splits)]
        scores = []

        for i in range(self.n_splits):
            X_test = folds_X[i]
            y_test = folds_y[i]

            X_train = np.vstack([folds_X[j] for j in range(self.n_splits) if j != i])
            y_train = np.hstack([folds_y[j] for j in range(self.n_splits) if j != i])

            # Cria uma cópia do modelo para evitar treino acumulado
            from sklearn.base import clone
            modelo = clone(model)
            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)

            mse = self._compute_score(y_test, y_pred)
            scores.append(mse)

        return scores

sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg,X, y)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))