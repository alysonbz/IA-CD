from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import confusion matrix e classification report
from sklearn.metrics import confusion_matrix, classification_report

# Carregar o dataset
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=6)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Gerar matriz de confusão e relatório de classificação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))