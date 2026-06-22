from dataset.pre_processamento import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)

X = all_batch[[f'feature_{i}' for i in range(1,129)]]
y = y.values.ravel()
scaler = StandardScaler()
X_ss = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size=0.2, random_state=42, stratify=y)

X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('SEM PCA')
print('Accuracy :', accuracy_score(y_test,y_pred))
print('Precision:', precision_score(y_test,y_pred,average='weighted'))
print('Recall   :', recall_score(y_test,y_pred,average='weighted'))
print('F1       :', f1_score(y_test,y_pred,average='weighted'))

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('KNN sem PCA')
plt.show()

pca = PCA()

X_pca = pca.fit_transform(X_ss)

variancia_acumulada = np.cumsum(
    pca.explained_variance_ratio_
)

n_comp = np.argmax(
    variancia_acumulada >= 0.95
) + 1

print('\nComponentes:', n_comp)

plt.figure(figsize=(8,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(0.95,color='red',linestyle='--',label=f'{n_comp}')
plt.legend()
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Acumulada')
plt.title('PCA')
plt.grid()

plt.show()

pca = PCA(n_components=n_comp)

X_train_pca = pca.fit_transform(X_train_ss)
X_test_pca = pca.transform(X_test_ss)

print(X_train.shape)
print(X_train_pca.shape)

knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca,y_train)

y_pred_pca = knn_pca.predict(
    X_test_pca
)

print('\nCOM PCA')
print('Accuracy :', accuracy_score(y_test,y_pred_pca))
print('Precision:', precision_score(y_test,y_pred_pca,average='weighted'))
print('Recall   :', recall_score(y_test,y_pred_pca,average='weighted'))
print('F1       :', f1_score(y_test,y_pred_pca,average='weighted'))

cm = confusion_matrix(y_test,y_pred_pca)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('KNN com PCA')
plt.show()

pca = PCA(n_components=2)
pca.fit(X_train)
transformado = pca.transform(X_train)
xs = transformado[:,0]
ys = transformado[:,1]
plt.scatter(xs,ys, c=y_train)
plt.show()