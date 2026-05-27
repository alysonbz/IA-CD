from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


k_values = []
accuracy_values = []


def treino_knn_minmax(X_train, X_test, y_train, y_test):

    print('Treinando modelo KNN com Min-Max:')

    melhor_acc = 0
    melhor_k = 0

    for k in range(1, 31):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)

        acc = accuracy_score(y_test, pred)

        # Salvando valores para o gráfico
        k_values.append(k)
        accuracy_values.append(acc)

        print(f'k = {k} | acurácia = {acc:.4f}')

        if acc > melhor_acc:
            melhor_acc = acc
            melhor_k = k

    print('Melhor k Min-Max:', melhor_k)
    print('Melhor acurácia:', melhor_acc)

    return melhor_acc, melhor_k