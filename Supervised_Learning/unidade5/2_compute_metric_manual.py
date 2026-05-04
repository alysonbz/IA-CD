from src.utils import process_diabetes

class Metrics:

    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
        # Counters for Class 1 (Positive)
        self.vp_c1, self.vn_c1, self.fp_c1, self.fn_c1 = 0, 0, 0, 0
        # Counters for Class 0 (Negative)
        self.vp_c0, self.vn_c0, self.fp_c0, self.fn_c0 = 0, 0, 0, 0

    def set_param_classe1(self):
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 1 and yt == 1: self.vp_c1 += 1
            elif yp == 0 and yt == 0: self.vn_c1 += 1
            elif yp == 1 and yt == 0: self.fp_c1 += 1
            elif yp == 0 and yt == 1: self.fn_c1 += 1

    def set_param_classe2(self):
        # For Class 0, the "Positive" is 0 and "Negative" is 1
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 0 and yt == 0: self.vp_c0 += 1 # True Positive for C0
            elif yp == 1 and yt == 1: self.vn_c0 += 1 # True Negative for C0
            elif yp == 0 and yt == 1: self.fp_c0 += 1 # False Positive for C0
            elif yp == 1 and yt == 0: self.fn_c0 += 1 # False Negative for C0

    def compute_acuraccy(self):
        total = self.vp_c1 + self.vn_c1 + self.fp_c1 + self.fn_c1
        return (self.vp_c1 + self.vn_c1) / total if total > 0 else 0

    def compute_recall_c1(self):
        denom = (self.vp_c1 + self.fn_c1)
        return self.vp_c1 / denom if denom > 0 else 0

    def compute_recall_c0(self):
        denom = (self.vp_c0 + self.fn_c0)
        return self.vp_c0 / denom if denom > 0 else 0

    def compute_precision_c1(self):
        denom = (self.vp_c1 + self.fp_c1)
        return self.vp_c1 / denom if denom > 0 else 0

    def compute_precision_c0(self):
        denom = (self.vp_c0 + self.fp_c0)
        return self.vp_c0 / denom if denom > 0 else 0

    def compute_f1_score_c1(self):
        p, r = self.compute_precision_c1(), self.compute_recall_c1()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    def compute_f1_score_c0(self):
        p, r = self.compute_precision_c0(), self.compute_recall_c0()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    def compute_confusion_matriz(self):
        # Format: [[VN, FP], [FN, VP]] for Class 1 perspective
        return [[self.vn_c1, self.fp_c1], [self.fn_c1, self.vp_c1]]

# Execution
y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print(f"Acurácia geral: {mt.compute_acuraccy():.2f}")
print(f"Recall classe 0: {mt.compute_recall_c0():.2f}")
print(f"Recall classe 1: {mt.compute_recall_c1():.2f}")
print(f"Precision classe 0: {mt.compute_precision_c0():.2f}")
print(f"Precision classe 1: {mt.compute_precision_c1():.2f}")
print(f"F1-score classe 0: {mt.compute_f1_score_c0():.2f}")
print(f"F1-score classe 1: {mt.compute_f1_score_c1():.2f}")
print(f"Matriz de confusão: {mt.compute_confusion_matriz()}")