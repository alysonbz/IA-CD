from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Obter as probabilidades, valores reais e predições
y_prob, y_test, y_pred = log_reg_diabetes()

# Calcular a AUC (área sob a curva ROC)
print(roc_auc_score(y_test, y_prob))

# Calcular a matriz de confusão
print(confusion_matrix(y_test, y_pred))

# Gerar o relatório de classificação
print(classification_report(y_test, y_pred))