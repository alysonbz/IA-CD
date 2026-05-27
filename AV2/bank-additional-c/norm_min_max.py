from sklearn.preprocessing import MinMaxScaler


def aplicar_min_max(X_train, X_test):

    scaler = MinMaxScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm