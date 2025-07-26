import matplotlib.pyplot as plt
from src.utils import log_reg_diabetes
# Import roc_curve
from sklearn.metrics import roc_curve

# Obter as probabilidades e y_test a partir da função fornecida
y_prob, y_test, _ = log_reg_diabetes()

# Gerar os valores da curva ROC: fpr (false positive rate), tpr (true positive rate), thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Linha de referência (classificador aleatório)
plt.plot([0, 1], [0, 1], 'k--')

# Plotar a curva ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()