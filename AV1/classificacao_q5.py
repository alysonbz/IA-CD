import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report



#Aplique `cross_val_score` (5 folds) com a melhor configuração possível que você determinou do seu classificador.
#Exiba: média, desvio padrão, matriz de confusão e `classification_report`.
#Interprete os resultados quantitativamente.


# Carregar os dados
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\voice.csv")
X = df.drop("label", axis=1)
y = df["label"].map({'male': 0, 'female': 1})

# Normalização
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Melhor K encontrado: 6
knn = KNeighborsClassifier(n_neighbors=6)

# Avaliação com cross_val_score (5 folds)
scores = cross_val_score(knn, X_scaled, y, cv=5)
print(f"Acurácia média (Validação - CV): {scores.mean():.4f}")
print(f"Desvio padrão : {scores.std():.4f}")

# Avaliação com cross_val_predict para matriz e relatório
y_pred = cross_val_predict(knn, X_scaled, y, cv=5)

# Matriz de confusão
cm = confusion_matrix(y, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# Classification report
report = classification_report(y, y_pred, target_names=["male", "female"])
print("\nRelatório de Classificação:")
print(report)