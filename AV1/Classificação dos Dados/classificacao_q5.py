import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 🔗 Carregar os dados
df = pd.read_csv('classificacao_ajustado.csv')

# 🎯 Definir variável alvo (inadimplente ou não)
df['target'] = (df['OVD_sum'] > 0).astype(int)

# 🎯 Definir X e y
X = df.drop(columns=['id', 'OVD_sum', 'target', 'update_date', 'report_date'])
y = df['target']

# ✅ Aplicando a melhor configuração encontrada anteriormente (K=5, StandardScaler)
pipeline_final = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

# 🔍 Cross-validation
scores = cross_val_score(pipeline_final, X, y, cv=5, scoring='accuracy')

print('Média da Acurácia:', np.mean(scores))
print('Desvio Padrão:', np.std(scores))

# 🔍 Predições para matriz de confusão e classification_report
y_pred = cross_val_predict(pipeline_final, X, y, cv=5)

# 📊 Matriz de confusão
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# 📄 Classification Report
report = classification_report(y, y_pred)
print('Classification Report:')
print(report)

# 📝 Interpretação Quantitativa
interpretacao = """
O modelo obteve altíssima acurácia (99.5%), o que indica ótimo desempenho geral.

A classe 0 (não inadimplente) foi classificada perfeitamente (100% de revocação).

A classe 1 (inadimplente) teve precisão de 100%, ou seja, nenhum cliente foi falsamente rotulado como inadimplente.

A revocação da classe 1 foi de 95.9%, o que mostra que poucos inadimplentes não foram identificados (apenas 38 falsos negativos).

A f1-score ponderada de 0.9953 confirma a excelente performance balanceada entre as classes.

Matriz de confusão:

O modelo acertou 7330 vezes que o cliente era não inadimplente (TN).

Não cometeu nenhum falso positivo (FP = 0). Isso significa que nenhum cliente foi erroneamente classificado como inadimplente.

Cometeu 38 falsos negativos (FN), ou seja, deixou de identificar 38 clientes que eram inadimplentes.

Acertou 882 vezes que o cliente era inadimplente (TP).
"""

print(interpretacao)
