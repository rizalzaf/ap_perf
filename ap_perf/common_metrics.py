from ap_perf.metric import PerformanceMetric, Confusion_Matrix, sqrt


# metrics
class Accuracy(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        return (C.tp + C.tn) / (C.all)  

accuracy_metric = Accuracy()
accuracy_metric.initialize()


class Precision(PerformanceMetric):       # Precision
    def define(self, C: Confusion_Matrix):
        return C.tp / C.pp

prec = Precision()
prec.initialize()
prec.enforce_special_case_positive()


class Recall(PerformanceMetric):       # Recall / Sensitivity
    def define(self, C: Confusion_Matrix):
        return C.tp / C.ap
  
rec = Recall()
rec.initialize()
rec.enforce_special_case_positive()


class Specificity(PerformanceMetric):       # Specificity
    def define(self, C: Confusion_Matrix):
        return C.tn / C.an
   
spec = Specificity()
spec.initialize()
spec.enforce_special_case_negative()


# F1
class F1Score(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        return (2 * C.tp) / (C.ap + C.pp)  

f1_score = F1Score()
f1_score.initialize()
f1_score.enforce_special_case_positive()


# metric with arguments
class FBeta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C: Confusion_Matrix):
        return ((1 + self.beta ** 2) * C.tp) / ( (self.beta ** 2) * C.ap + C.pp)  

f2_score = FBeta(2)
f2_score.initialize()
f2_score.enforce_special_case_positive()

fhalf_score = FBeta(0.5)
fhalf_score.initialize()
fhalf_score.enforce_special_case_positive()


# Geometric Mean of Prec and Rec
class GM_PrecRec(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        return C.tp / sqrt(C.ap * C.pp)  

gpr = GM_PrecRec()
gpr.initialize()
gpr.enforce_special_case_positive()


# informedness
class Informedness(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        return C.tp / C.ap + C.tn / C.an - 1

inform = Informedness()
inform.initialize()
inform.enforce_special_case_positive()
inform.enforce_special_case_negative()


class Kappa(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        pe = (C.ap * C.pp + C.an * C.pn) / C.all**2
        num = (C.tp + C.tn) / C.all - pe
        den = 1 - pe
        return num / den  

kappa = Kappa()
kappa.initialize()
kappa.enforce_special_case_positive()
kappa.enforce_special_case_negative()


class MCC(PerformanceMetric):
    def define(self, C: Confusion_Matrix):
        num = C.tp / C.all - (C.ap * C.pp) / C.all**2
        den = sqrt(C.ap * C.pp * C.an * C.pn) / C.all**2
        return num / den

mcc = MCC()
mcc.initialize()
mcc.enforce_special_case_positive()
mcc.enforce_special_case_negative()



########## METRIC WITH CONSTRAINTS

# precision given recall
class PrecisionGvRecall(PerformanceMetric):
    def __init__(self, th):
        self.th = th

    def define(self, C: Confusion_Matrix):
        return C.tp / C.pp

    def constraint(self, C: Confusion_Matrix):
        return C.tp / C.ap >= self.th


precision_gv_recall_80 = PrecisionGvRecall(0.8)
precision_gv_recall_80.initialize()
precision_gv_recall_80.enforce_special_case_positive()
precision_gv_recall_80.set_cs_special_case_positive(True)

precision_gv_recall_60 = PrecisionGvRecall(0.6)
precision_gv_recall_60.initialize()
precision_gv_recall_60.enforce_special_case_positive()
precision_gv_recall_60.set_cs_special_case_positive(True)

precision_gv_recall_95 = PrecisionGvRecall(0.95)
precision_gv_recall_95.initialize()
precision_gv_recall_95.enforce_special_case_positive()
precision_gv_recall_95.set_cs_special_case_positive(True)


# recall given precision
class RecallGvPrecision(PerformanceMetric):
    def __init__(self, th):
        self.th = th

    def define(self, C: Confusion_Matrix):
        return C.tp / C.pp

    def constraint(self, C: Confusion_Matrix):
        return C.tp / C.ap >= self.th
   

recal_gv_precision_80 = RecallGvPrecision(0.8)
recal_gv_precision_80.initialize()
recal_gv_precision_80.enforce_special_case_positive()
recal_gv_precision_80.set_cs_special_case_positive(True)


class PrecisionGvRecallSpecificity(PerformanceMetric):        # precision given recall >= th1 and specificity >= th2
    def __init__(self, th1, th2):
        self.th1 = th1
        self.th2 = th2

    def define(self, C: Confusion_Matrix):
        return C.tp / C.pp

    def constraint(self, C: Confusion_Matrix):
        return [C.tp / C.ap >= self.th1,
            C.tn / C.an >= self.th2]

precision_gv_recall_spec = PrecisionGvRecallSpecificity(0.8, 0.8)
precision_gv_recall_spec.initialize()
precision_gv_recall_spec.enforce_special_case_positive()
precision_gv_recall_spec.set_cs_special_case_positive([True, False])
precision_gv_recall_spec.set_cs_special_case_negative([False, True])

