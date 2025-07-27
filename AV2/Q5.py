from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

cliente_df['Alvo'] = cliente_df['Cluster']  # Usar cluster como pseudo-label

# Cenário 1: sem redução
X_train, X_test, y_train, y_test = train_test_split(X_scaled, cliente_df['Alvo'], test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Sem Redução:\n", classification_report(y_test, y_pred))

# Cenário 2: com PCA
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, cliente_df['Alvo'], test_size=0.3, random_state=42)
clf.fit(X_pca_train, y_train)
y_pred = clf.predict(X_pca_test)
print("Com PCA:\n", classification_report(y_test, y_pred))

# Cenário 3: com T-SNE (menos comum, mas comparativo)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, cliente_df['Alvo'], test_size=0.3, random_state=42)
clf.fit(X_tsne_train, y_train)
y_pred = clf.predict(X_tsne_test)
print("Com T-SNE:\n", classification_report(y_test, y_pred))