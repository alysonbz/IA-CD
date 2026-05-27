from sklearn.preprocessing import StandardScaler


def aplicar_z_score(X_train, X_test):

    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm