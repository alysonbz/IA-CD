from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def treinar_knn_log(X_train, X_test, y_train, y_test):

    melhores_resultados = []

    melhor_acc = 0
    melhor_k = 0

    for k in range(1, 31):

        modelo = KNeighborsClassifier(n_neighbors=k)

        modelo.fit(X_train, y_train)

        previsoes = modelo.predict(X_test)

        acc = accuracy_score(y_test, previsoes)

        melhores_resultados.append(acc)

        print(f'Log | k = {k} | acc = {acc:.4f}')

        if acc > melhor_acc:
            melhor_acc = acc
            melhor_k = k

    print('Melhor resultado Log')
    print(f'Melhor k: {melhor_k}')
    print(f'Melhor acurácia: {melhor_acc:.4f}')

    return melhor_acc, melhor_k