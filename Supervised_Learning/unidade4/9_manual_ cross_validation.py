import numpy as np
from src.utils import load_sales_clean_dataset


class LinearRegressionManual:

    def __init__(self):
        self.a = 0
        self.b = 0

    def fit(self, X, y):
        X = X.flatten()

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.a = numerator / denominator
        self.b = y_mean - self.a * x_mean

    def predict(self, X):
        X = X.flatten()
        return self.a * X + self.b


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X, y):
        obj.fit(X[0], y[0])

        y_pred = obj.predict(X[1])

        ss_res = np.sum((y[1] - y_pred) ** 2)
        ss_tot = np.sum((y[1] - np.mean(y[1])) ** 2)

        score = 1 - (ss_res / ss_tot)

        return score

    def cross_val_score(self, obj, X, y):

        scores = []

        # parte 1: dividir o dataset X em n_splits vezes
        tamanho = len(X)
        indices = np.arange(tamanho)

        partes = np.array_split(indices, self.n_splits)

        for i in range(self.n_splits):
            test_index = partes[i]

            train_index = np.concatenate(
                [partes[j] for j in range(self.n_splits) if j != i]
            )

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset
            score = self._compute_score(
                obj,
                [X_train, X_test],
                [y_train, y_test]
            )

            # append na lista scores cada valor obtido na parte 2
            scores.append(score)

        # parte 3 - retornar a lista scores
        return np.array(scores)


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

reg = LinearRegressionManual()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg, X, y)

# Print scores
print("Scores:")
for score in cv_scores:
    print(f"{score:.6f}")

# Print the mean
print("\nMean:")
print(f"{np.mean(cv_scores):.6f}")

# Print the standard deviation
print("\nStandard deviation:")
print(f"{np.std(cv_scores):.6f}")