import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar dados
df = pd.read_csv('dataset/marketing_campaign_with_clusters.csv')

# Variável alvo (exemplo: Response)
df['Response'] = (df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']) > 0
df['Response'] = df['Response'].astype(int)

y = df['Response']
X = df.drop(columns=[
    'Response',
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
    'AcceptedCmp4', 'AcceptedCmp5'
])

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cenário 1 - Somente normalizado
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf1 = RandomForestClassifier(random_state=42, class_weight='balanced')
clf1.fit(X_train1, y_train1)
print("Normalizado:")
print(classification_report(y_test1, clf1.predict(X_test1)))

# Cenário 2 - PCA
X_pca = PCA(n_components=10).fit_transform(X_scaled)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pca, y, test_size=0.2, random_state=42)
clf2 = RandomForestClassifier(random_state=42, class_weight='balanced')
clf2.fit(X_train2, y_train2)
print("PCA:")
print(classification_report(y_test2, clf2.predict(X_test2)))

# Cenário 3 - T-SNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_tsne, y, test_size=0.2, random_state=42)
clf3 = RandomForestClassifier(random_state=42, class_weight='balanced')
clf3.fit(X_train3, y_train3)
print("T-SNE:")
print(classification_report(y_test3, clf3.predict(X_test3)))
