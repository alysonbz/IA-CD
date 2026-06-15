from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from src.utils import load_fish_dataset

fish = load_fish_dataset()

labels = fish['specie']
samples = fish.drop(['specie'], axis=1)

labels_encoded = LabelEncoder().fit_transform(labels)

scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_samples)

X_train, X_test, y_train, y_test = train_test_split(
pca_features, labels_encoded, test_size=0.25, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))