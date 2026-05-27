from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score
)

import matplotlib.pyplot as plt


def avaliar_modelo(modelo, X_test, y_test, nome_modelo):

    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f'\n===== {nome_modelo} =====')
    print(f'Acurácia: {acc:.4f}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # MATRIZ DE CONFUSÃO

    matriz = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz
    )

    disp.plot(cmap='Blues')

    plt.title(f'Matriz de Confusão - {nome_modelo}')

    plt.show()