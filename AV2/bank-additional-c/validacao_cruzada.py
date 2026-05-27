from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def validacao_cruzada(X_train, y_train, k):

    print('Executando validação cruzada...')

    modelo = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    print('Scores:', scores)
    print('Média:', scores.mean())