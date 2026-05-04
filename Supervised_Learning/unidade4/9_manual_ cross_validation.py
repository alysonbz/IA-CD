import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def __init__(self,n_splits):

        self.n_splits = n_splits

    def _compute_score(self, model, X_train, X_test, y_train, y_test):
        # Treina o modelo
        model.fit(X_train, y_train)

        # Retorna o R²
        return model.score(X_test, y_test)

    def cross_val_score(self,obj,X, y):

        scores = []

        n = len(X)
        fold_size = n // self.n_splits

        indices = np.arange(n)

        # parte 1: dividir o dataset X em n_splits vezes
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size

            # Índices de teste
            test_idx = indices[start:end]

            # Índices de treino (tudo menos o teste)
            train_idx = np.concatenate((indices[:start], indices[end:]))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset
        #appendar na lista scores cada valor obtido na parte 2
        score = self._compute_score(obj, X_train, X_test, y_train, y_test)


        scores.append(score)

        #parte 3 - retornar a lista de scores
        return scores


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
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

