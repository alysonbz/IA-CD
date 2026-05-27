import matplotlib.pyplot as plt

from treino_min_max import k_values
from treino_min_max import accuracy_values


def grafico_k():

    print('Gerando gráfico de acurácia por valor de k:')

    plt.figure(figsize=(10,6))

    plt.plot(k_values, accuracy_values, marker='o')

    plt.title('Acurácia do KNN por valor de k')
    plt.xlabel('Valor de k')
    plt.ylabel('Acurácia')

    plt.grid(True)

    plt.show()