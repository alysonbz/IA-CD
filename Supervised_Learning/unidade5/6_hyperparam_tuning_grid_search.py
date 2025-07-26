import numpy as np

# Import Lasso
from sklearn.linear_model import Lasso

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import KFold
from sklearn.model_selection import KFold

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

from src.utils import load_diabetes_clean_dataset

# Carregar o dataset
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['glucose'], axis=1)
y = diabetes_df['glucose'].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Inicializar o modelo Lasso
lasso = Lasso()

# Inicializar o KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir a grade de parâmetros (valores de alpha para testar)
param_grid = {"alpha": np.linspace(0.01, 1, 100)}

# Instanciar o GridSearchCV com validação cruzada
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Treinar o modelo
lasso_cv.fit(X_train, y_train)

# Exibir os melhores resultados
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))