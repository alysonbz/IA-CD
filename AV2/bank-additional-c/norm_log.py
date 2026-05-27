import numpy as np


def normalizacao_log(X_train, X_test):

    print('Aplicando transformação Log:')

    # Copiando os dados
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()

    # Ajustando valores negativos
    for coluna in X_train_log.columns:

        menor_valor = min(
            X_train_log[coluna].min(),
            X_test_log[coluna].min()
        )

        # Se existir valor negativo
        if menor_valor <= -1:

            ajuste = abs(menor_valor) + 1

            X_train_log[coluna] = X_train_log[coluna] + ajuste
            X_test_log[coluna] = X_test_log[coluna] + ajuste

    # Aplicando log
    X_train_log = np.log1p(X_train_log)
    X_test_log = np.log1p(X_test_log)

    print('Transformação Log concluída!')

    print('Verificando NaN:')
    print(np.isnan(X_train_log).sum())

    return X_train_log, X_test_log