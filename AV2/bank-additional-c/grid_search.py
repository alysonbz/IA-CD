from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier



def executar_grid_search(X_train, y_train):

    parametros = {
        'n_neighbors': range(1, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    modelo = KNeighborsClassifier()

    grid = GridSearchCV(
        modelo,
        parametros,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    print('Melhores parâmetros:')
    print(grid.best_params_)

    print('Melhor score:')
    print(grid.best_score_)

    return grid