from ap_perf.metric import *
import numpy as np
import pytest
import numpy.testing as npt

tp = CM_Entity(CM_Type.TP, CM_Category.CELL) 
fp = CM_Entity(CM_Type.FP, CM_Category.CELL) 
fn = CM_Entity(CM_Type.FN, CM_Category.CELL) 
tn = CM_Entity(CM_Type.TN, CM_Category.CELL) 
ap = CM_Entity(CM_Type.AP, CM_Category.ACTUAL_SUM) 
an = CM_Entity(CM_Type.AN, CM_Category.ACTUAL_SUM) 
pp = CM_Entity(CM_Type.PP, CM_Category.PREDICTION_SUM)
pn = CM_Entity(CM_Type.PN, CM_Category.PREDICTION_SUM)
al = CM_Entity(CM_Type.ALL, CM_Category.ALL_SUM)

def test_simple():
    assert str(2 * tp) == "2 TP"
    assert str(tp + fp) == "PP"
    assert str(2 * tp / (ap + pp)) == "(2 TP) / ((AP) + (PP))" 

def test_complex():
    kappa = ((tp + tn) / al - (ap * pp + an * pn) / al**2) / (1 - (ap * pp + an * pn) / al**2)
    str_kappa = "((((TP) + (TN)) / (ALL)) + (-((((AP) * (PP)) + ((AN) * (PN))) / ((ALL)**2)))) / ((1) + (-((((AP) * (PP)) + ((AN) * (PN))) / ((ALL)**2))))"
    assert str(kappa) == str_kappa

def test_info():
    acc = (tp + tn) / (ap + an)
    assert acc.needs_adv_sum_marg == False

    prec = tp / pp
    assert prec.needs_pred_sum_marg == True
    assert prec.needs_adv_sum_marg == False

    gm = tp / (sqrt(pp * ap))
    assert gm.needs_adv_sum_marg == True
    assert gm.needs_pred_sum_marg == True
    assert gm.is_linear_tp_tn == True
    

    kappa = ((tp + tn) / al - (ap * pp + an * pn) / al**2) / (1 - (ap * pp + an * pn) / al**2)
    assert kappa.needs_pred_sum_marg == True
    assert kappa.needs_adv_sum_marg == True
    assert kappa.is_linear_tp_tn == True

    gm_err = tp / (sqrt(pp * tn))
    assert gm_err.needs_pred_sum_marg == True
    assert gm_err.needs_adv_sum_marg == False
    assert gm_err.is_linear_tp_tn == False

def test_compute_metric():
    class Prec(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return C.tp / C.pp   

    class F1_score(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return (2 * C.tp) / (C.ap + C.pp)   
    
    class Kappa(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            num = (C.tp + C.tn) / C.all - (C.ap * C.pp + C.an * C.pn) / C.all**2
            den = 1 - (C.ap * C.pp + C.an * C.pn) / C.all**2
            return num / den   

    prec_fn = lambda yhat, y : (np.dot(yhat, y) / np.sum(yhat))
    f1_fn = lambda yhat, y : (2 * np.dot(yhat, y) / (np.sum(yhat) + np.sum(y)))
    def kappa_fn(yhat, y):
        n = len(y)
        num = ((np.dot(yhat, y) + np.dot(1-yhat, 1-y)) / n - 
            (np.sum(y) * np.sum(yhat) + np.sum(1-y) * np.sum(1-yhat)) / n**2 )
        den = 1 - (np.sum(y) * np.sum(yhat) + np.sum(1-y) * np.sum(1-yhat)) / n**2
        return num / den

    y = np.random.randint(2, size=100)
    yhat = np.random.randint(2, size=100)

    prec = Prec()
    prec.initialize()
    assert prec.compute_metric(yhat, y) == prec_fn(yhat, y)

    f1 = F1_score()
    f1.initialize()
    assert f1.compute_metric(yhat, y) == f1_fn(yhat, y)

    kappa = Kappa()
    kappa.initialize()
    assert kappa.compute_metric(yhat, y) == kappa_fn(yhat, y)

def test_invalid_metric():

    class Metric1(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return C.tp * C.tn / sqrt(C.ap * C.pp)

    class Metric2(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return C.ap / C.all

    class Metric3(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return (C.tp / C.all) * (1 - C.fp) / (C.pp)

    class Metric4(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return C.tp / (C.ap + C.pp + C.fp)
    
    with pytest.raises(UnsupportedMetric):
        Metric1().initialize()

    with pytest.raises(UnsupportedMetric):
        Metric2().initialize()

    with pytest.raises(UnsupportedMetric):
        Metric3().initialize()

    with pytest.raises(UnsupportedMetric):
        Metric4().initialize()
    

 
def test_generate_constraint():

    def f1_fn(y):
        n = len(y)
        iq = np.sum(y == 1)

        ks = np.linspace(1, n, n)
        B = np.zeros((n,n))
        c = 0.0
        
        for ip in range(n):
            B[:,ip] = ( 2 * y ) / (ip + iq + 1)

        if iq == 0:        
            B += (-1 / ks)
            c += 1.0

        return B, c

    class Prec_F1_score(PerformanceMetric):
        def define(self, C: Confusion_Matrix):
            return C.tp / C.pp   

        def constraint(self, C: Confusion_Matrix):
            return (2 * C.tp) / (C.ap + C.pp) >= 0.6

    prec_f1 = Prec_F1_score()
    prec_f1.initialize()
    prec_f1.enforce_special_case_positive()
    prec_f1.set_cs_special_case_positive(True)

    y = np.random.randint(2, size=20)
    prec_f1._PerformanceMetric__generate_constraints_on_p(y)

    B, c = f1_fn(y)
    npt.assert_almost_equal(B, prec_f1.B_list[0])
    npt.assert_almost_equal(c, prec_f1.c_list[0])

    # if y == zeros
    y = np.zeros(20)
    prec_f1._PerformanceMetric__generate_constraints_on_p(y)

    B, c = f1_fn(y)
    npt.assert_almost_equal(B, prec_f1.B_list[0])
    npt.assert_almost_equal(c, prec_f1.c_list[0])

    # if y == ones
    y = np.ones(20)
    prec_f1._PerformanceMetric__generate_constraints_on_p(y)

    B, c = f1_fn(y)
    npt.assert_almost_equal(B, prec_f1.B_list[0])
    npt.assert_almost_equal(c, prec_f1.c_list[0])

