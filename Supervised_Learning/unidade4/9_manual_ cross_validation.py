import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression

class KFold:

   def __init__(self,n_splits):
       self.n_splits = n_splits

   def train_test_split(self,X, y):
       X_train, X_test, y_train, y_test = None
       return X_train, X_test, y_train, y_test

   def _compute_score(self,X,y,obj):
       pred = obj.predict(X)
       var_pred = np.sum(np.square((pred - np.mean(y))))
       var_data = np.sum(np.square((y - np.mean(y))))
       r_squared = np.divide(var_pred, var_data)
       return r_squared

   def cross_val_score(self,obj,X, y):
        scores = []
        x_train_n = []
        y_train_n = []
        x_test_n = []
        y_test_n = []

        # Parte 0: embaralhar
        X = np.random.shuffle(X)
        y = np.random.shuffle(y)

        # parte 1: dividir o dataset X em n_splits vezes
        for n in range(0, self.n_splits):
            X_train, X_test, y_train, y_test = self.train_test_split(X,y)
            x_train_n.append(X_train)
            x_test_n.append(X_test)
            y_train_n.append(y_train)
            y_test_n.append(y_test)

        # parte 2: Calcular a métrica score para subset dividida na parte 1. Chamar a função _compute_score para cada subset

        #appendar na lista scores cada valor obtido na parte 2

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

