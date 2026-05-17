from src.utils import process_diabetes
import numpy as np

class Metrics:

    def __init__(self, y_pred, y_test):
        self.vp_c1 = 0
        self.vn_c1 = 0
        self.fp_c1 = 0
        self.fn_c1 = 0
        self.vp_c0 = 0
        self.vn_c0 = 0
        self.fp_c0 = 0
        self.fn_c0 = 0

    def set_param_classe1(self):

        for yp, yt in zip(y_pred, y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 = self.vp_c1 + 1
            if yp == 1 and yt == 0:
                self.fp_c1 = self.fp_c1 + 1
            if yp == 0 and yt == 1:
                self.fn_c1 = self.fn_c1 + 1
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def set_param_classe0(self):
        for yp, yt in zip(y_pred, y_test):
            if yp == 0 and yt == 0:
                self.vp_c0 = self.vp_c0 + 1
            if yp == 0 and yt == 1:
                self.fp_c0 = self.fp_c0 + 1
            if yp == 1 and yt == 0:
                self.fn_c0 = self.fn_c0 + 1
            if yp == 1 and yt == 1:
                self.vn_c0 = self.vn_c0 + 1

    def compute_acuraccy(self):
        acc = (self.vp_c1+self.vn_c1)/(self.vp_c1+self.vn_c1+self.fp_c1+self.fn_c1)
        return acc

    def compute_recall_c1(self):
        recall = self.vp_c1/(self.vp_c1+self.fn_c1)
        return recall

    def compute_recall_c0(self):
        recall = self.vp_c0 / (self.vp_c0 + self.fn_c0)
        return recall

    def compute_precision_c1(self):
        precision = self.vp_c1/(self.vp_c1+self.fp_c1)
        return precision

    def compute_precision_c0(self):
        precision = self.vp_c0 / (self.vp_c0 + self.fp_c0)
        return precision

    def compute_f1_score_c1(self):
        f1_score = (2*(self.compute_precision_c1()*self.compute_recall_c1()))/(self.compute_precision_c1()+self.compute_recall_c1())
        return f1_score

    def compute_f1_score_c0(self):
        f1_score = (2*(self.compute_precision_c0()*self.compute_recall_c0()))/(self.compute_precision_c0()+self.compute_recall_c0())
        return f1_score

    def compute_confusion_matriz(self):
        matriz = np.array([[self.vp_c1,self.fn_c1],
                           [self.fp_c1,self.vn_c1]])
        return matriz


y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe0()


print("acurácia geral:", mt.compute_acuraccy())
#
print("recall classe 0: ", mt.compute_recall_c0())
#
print("recall classe 1:", mt.compute_recall_c1())
#
print("precision classe 0: ", mt.compute_precision_c0())
#
print("precision classe 1:", mt.compute_precision_c1())
#
print("F1-score classe 0:", mt.compute_f1_score_c0())
#
print("F1-score classe 1:", mt.compute_f1_score_c1())
#
print("Matriz de confusão\n", mt.compute_confusion_matriz())
#