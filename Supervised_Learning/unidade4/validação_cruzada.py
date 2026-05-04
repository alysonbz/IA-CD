import numpy as np


def manual_cross_validation(X, y, k=5):
    # 1. Conversão para arrays e Shuffle (Embaralhamento)
    X = np.array(X)
    y = np.array(y)

    indices = np.arange(len(X))
    np.random.seed(42)  # Semente fixa para resultados reproduzíveis
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 2. Divisão dos Folds
    fold_size = len(X) // k
    scores = []

    print(f"--- Executando Validação Cruzada Manual (k={k}) ---")

    for i in range(k):
        # 3. Definição dos índices de teste/validação
        start, end = i * fold_size, (i + 1) * fold_size

        X_test = X[start:end]
        y_test = y[start:end]

        # O treino é tudo o que NÃO é teste
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        # 4. Cálculo da Acurácia (Exemplo: Usando a lógica do Vizinho Mais Próximo)
        # Aqui você pode substituir pela chamada do seu modelo específico
        correct_predictions = 0
        for j in range(len(X_test)):


            # Simulando uma lógica de predição real para o exercício:
            distancias = np.linalg.norm(X_train - X_test[j], axis=1)
            vizinho_mais_proximo = y_train[np.argmin(distancias)]

            if vizinho_mais_proximo == y_test[j]:
                correct_predictions += 1

        accuracy = correct_predictions / len(y_test)
        scores.append(accuracy)
        print(f"Fold {i + 1}: Acurácia = {accuracy:.4f}")

    # 5. Resultados Consolidados
    print("-" * 40)
    print(f"Média das Acurácias: {np.mean(scores):.4f}")
    print(f"Desvio Padrão: {np.std(scores):.4f}")


# Exemplo de execução com dados de teste
if __name__ == "__main__":
    # Criando um dataset pequeno de exemplo
    from sklearn.datasets import load_iris  # Apenas para pegar os dados, não o CV

    data = load_iris()
    manual_cross_validation(data.data, data.target, k=5)