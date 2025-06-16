# 1. Aplique cross_val_score (5 folds) com a melhor configuração possível que você determinou do seu classificador.

from sklearn.model_selection import cross_val_score, cross_val_predict

df_2 = pd.read_csv("/content/diabetes.csv")

x = df_2.drop("Outcome", axis=1)
y = df_2["Outcome"]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])


scores = cross_val_score(pipe, x, y, cv=5, scoring='accuracy')

print(f"Acurácia: {scores}")

##

# 2. Exiba: média, desvio padrão, matriz de confusão e classification_report.

from sklearn.metrics import classification_report, confusion_matrix
scores = cross_val_score(pipe, x, y, cv=5, scoring='accuracy')

print(f"Média: {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

# Previsões com cross_val_predict
y_pred = cross_val_predict(pipe, x, y, cv=5)

# Matriz de confusão
conf_matrix = confusion_matrix(y, y_pred)
print("\nMatriz de Confusão:")
print(conf_matrix)

# Classification report
report = classification_report(y, y_pred)
print("\nRelatório de Classificação:")
print(report)

##

