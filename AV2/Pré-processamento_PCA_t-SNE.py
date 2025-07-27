# ===========================================
# Pré-processamento PCA/t-SNE para Classificação
# ===========================================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Target: gender
y = df['gender'].map({'female': 0, 'male': 1})  # binário
X = df[['math score', 'reading score', 'writing score']]

# 1. Cenário Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Cenário PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Cenário t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Função para avaliar
def evaluate_model(X, y, model, name):
    acc = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))
    f1 = np.mean(cross_val_score(model, X, y, cv=5, scoring='f1'))
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

# Modelos
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

print("Logistic Regression:")
evaluate_model(X_scaled, y, logreg, "Normalização")
evaluate_model(X_pca, y, logreg, "PCA")
evaluate_model(X_tsne, y, logreg, "t-SNE")

print("\nRandom Forest:")
evaluate_model(X_scaled, y, rf, "Normalização")
evaluate_model(X_pca, y, rf, "PCA")
evaluate_model(X_tsne, y, rf, "t-SNE")
