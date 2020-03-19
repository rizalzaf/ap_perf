from enum import Enum
import math
import numpy as np
import abc

# Type of the entity in the cnfusion matrix
class CM_Type(Enum):
    TP = 1
    FP = 2
    FN = 3
    TN = 4
    AP = 5
    AN = 6
    PP = 7
    PN = 8
    ALL = 9

class CM_Category(Enum):
    CELL = 1
    ACTUAL_SUM = 2
    PREDICTION_SUM = 3
    ALL_SUM = 4

# for storing constants
class ConstantOverPQ():
    def __init__(self):
        self.cPQ = None
        self.cPQ0 = None
        self.cP0Q = None
        self.cP0Q0 = None
        self.c = None

    def is_constant(self):
        if (self.cPQ is None and self.cPQ0 is None and self.cP0Q is None and 
            self.cP0Q0 is None and self.c is not None):
            return True
        else:
            return False

    def __mul__(self, other):
        result = ConstantOverPQ()
        if self.cPQ is not None: result.cPQ = other * self.cPQ
        if self.cPQ0 is not None: result.cPQ0 = other * self.cPQ0
        if self.cP0Q is not None: result.cP0Q = other * self.cP0Q
        if self.cP0Q0 is not None: result.cP0Q0 = other * self.cP0Q0
        if self.c is not None: result.c = other * self.c
        return result

    @staticmethod
    def add_none(x, y):
        if x is None and y is None:
            return None
        elif x is None:
            return y
        elif y is None:
            return x
        else:
            return x + y

    def __add__(self, other):
        result = ConstantOverPQ()
        result.cPQ = ConstantOverPQ.add_none(self.cPQ, other.cPQ)
        result.cPQ0 = ConstantOverPQ.add_none(self.cPQ0, other.cPQ0)
        result.cP0Q = ConstantOverPQ.add_none(self.cP0Q, other.cP0Q)
        result.cP0Q0 = ConstantOverPQ.add_none(self.cP0Q0, other.cP0Q0)
        result.c = ConstantOverPQ.add_none(self.c, other.c)
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)


# exception
class OperationUndefined(Exception):
    pass

class UnsupportedMetric(Exception):
    pass

class InputMismatch(Exception):
    pass


# Entity in the confusion matrix
class CM_Entity:
    def __init__(self, _type, category = CM_Category.CELL):
        self._type = _type 
        self.category = category

    def __pos__(self):
        return EXPR_UnaryEntity(self, 1)

    def __neg__(self):
        return EXPR_UnaryEntity(self, -1)    

    def __add__(self, other):      
        if isinstance(other, CM_Entity):
            if (self._type == CM_Type.TP and other._type == CM_Type.FP) or \
                (self._type == CM_Type.FP and other._type == CM_Type.TP):
                return CM_Entity(CM_Type.PP, CM_Category.PREDICTION_SUM)
            elif (self._type == CM_Type.TP and other._type == CM_Type.FN) or \
                (self._type == CM_Type.FN and other._type == CM_Type.TP):
                return CM_Entity(CM_Type.AP, CM_Category.ACTUAL_SUM) 
            elif (self._type == CM_Type.TN and other._type == CM_Type.FN) or \
                (self._type == CM_Type.FN and other._type == CM_Type.TN):
                return CM_Entity(CM_Type.PN, CM_Category.PREDICTION_SUM)
            elif (self._type == CM_Type.TN and other._type == CM_Type.FP) or \
                (self._type == CM_Type.FP and other._type == CM_Type.TN):
                return CM_Entity(CM_Type.AN, CM_Category.ACTUAL_SUM)
            elif (self._type == CM_Type.AP and other._type == CM_Type.AN) or \
                (self._type == CM_Type.AN and other._type == CM_Type.AP) or \
                (self._type == CM_Type.PP and other._type == CM_Type.PN) or \
                (self._type == CM_Type.PN and other._type == CM_Type.PP):
                return CM_Entity(CM_Type.ALL, CM_Category.ALL_SUM)
            else:
                return EXPR_Addition(EXPR_UnaryEntity(self), EXPR_UnaryEntity(other))    
        else:
            return EXPR_Addition(EXPR_UnaryEntity(self), other)
        
    def __sub__(self, other): 
        return self.__add__(-other)

    def __radd__(self, other):    
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryIdentity(other) + self
        else:
            return self + other
    
    def __rsub__(self, other):         
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryIdentity(other) - self
        else:
            return -self + other
    
    def __mul__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryEntity(self, other)
        elif isinstance(other, CM_Expression):
            return EXPR_Multiplication(EXPR_UnaryEntity(self), other)
        elif isinstance(other, CM_Entity):
            return EXPR_Multiplication(EXPR_UnaryEntity(self), EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __truediv__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryEntity(self, 1/other)
        elif isinstance(other, CM_Expression):
            return EXPR_Fraction(EXPR_UnaryEntity(self), other)
        elif isinstance(other, CM_Entity):
            return EXPR_Fraction(EXPR_UnaryEntity(self), EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" / operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_Power(EXPR_UnaryEntity(self), other)
        else:
            raise OperationUndefined(" ** operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __ge__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_Constraint(EXPR_UnaryEntity(self), other)
        else:
            raise OperationUndefined(" ** operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __repr__(self):
        return str(self._type.name)


# Expression that combine entities     
class CM_Expression:
    def __init__(self):
        # default values for expression
        self.is_linear_tp_tn = True         # is it linear w.r.t TP and TN
        self.depends_cell_cm = False        # does it depend on the cell of confusion matrix: tp, tn, fp, fn 
        self.depends_actual_sum = False     # does it depend on actual sum statistics: AP & AN
        self.depends_predicted_sum = False  # does it depend on predicted sum statistics: PP & PN
        self.is_constant = False            # does it contain only numbers or {ALL} entity
        self.needs_adv_sum_marg = False     # does it need 'sum'-marginal of the adversary to compute 
        self.needs_pred_sum_marg = False    # does it need 'sum'-marginal of the predictor to compute
        self.is_constraint = False          # is it a constraint (contains '>=')

    def __neg__(self):
        return EXPR_UnaryExpr(self, -1)
    
    def __add__(self, other):   
        if (type(other) == int or type(other) == float):
            return EXPR_Addition(self, EXPR_UnaryIdentity(other))
        elif isinstance(other, CM_Entity):
            return EXPR_Addition(self, EXPR_UnaryEntity(other))
        else:
            return EXPR_Addition(self, other)
        
    def __sub__(self, other): 
        return self.__add__(-other)

    def __radd__(self, other):    
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryIdentity(other) + self
        else:
            return self + other
    
    def __rsub__(self, other):         
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryIdentity(other) - self
        else:
            return -self + other

    def __mul__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryExpr(self, other)
        elif isinstance(other, CM_Expression):
            return EXPR_Multiplication(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Multiplication(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __truediv__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryExpr(self, 1/other)
        elif isinstance(other, CM_Expression):
            return EXPR_Fraction(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Fraction(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" / operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_Power(self, other)
        else:
            raise OperationUndefined(" ** operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __ge__(self, other):
        if (type(other) == int or type(other) == float):
            if self.is_constraint == False:
                return EXPR_Constraint(self, other)
            else:
                raise OperationUndefined("Left hand side of '>=' operator must not contains '>='")    
        else:
            raise OperationUndefined(" ** operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")
    
    def info(self):
        print("is_linear_tp_tn : ", self.is_linear_tp_tn)
        print("depends_cell_cm : ", self.depends_cell_cm)
        print("depends_actual_sum : ", self.depends_actual_sum)
        print("depends_predicted_sum : ", self.depends_predicted_sum)
        print("is_constant : ", self.is_constant)
        print("needs_adv_sum_marg : ", self.needs_adv_sum_marg)
        print("needs_pred_sum_marg : ", self.needs_pred_sum_marg)

    @abc.abstractmethod
    def compute_value(self, C_val):
        pass

    @abc.abstractmethod
    def compute_scaling(self, m, info):
        # m is n + 1, since the index are: 0,...,n
        # should return an instance of ConstantOverPQ
        pass


class EXPR_UnaryEntity(CM_Expression):
    def __init__(self, entity, multiplier = 1):
        self.entity = entity
        self.multiplier = multiplier

        super().__init__()
        if self.entity.category == CM_Category.CELL:
            self.depends_cell_cm = True
        elif self.entity.category == CM_Category.ACTUAL_SUM:
            self.depends_actual_sum = True
        elif self.entity.category == CM_Category.PREDICTION_SUM:
            self.depends_predicted_sum = True
        elif self.entity.category == CM_Category.ALL_SUM:
            self.is_constant = True

    def __pos__(self):
        return EXPR_UnaryEntity(self.entity, self.multiplier)

    def __neg__(self):
        return EXPR_UnaryEntity(self.entity, -self.multiplier)
    
    def __mul__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryEntity(self.entity, self.multiplier * other)
        elif isinstance(other, CM_Expression):
            return EXPR_Multiplication(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Multiplication(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __truediv__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryEntity(self.entity, self.multiplier / other)
        elif isinstance(other, CM_Expression):
            return EXPR_Fraction(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Fraction(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")                    

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __repr__(self):
        if self.multiplier == 1:
            return str(self.entity)
        elif self.multiplier == -1:
            return "-" + str(self.entity)
        else:
            return str(self.multiplier) + " " + str(self.entity)
    
    def compute_value(self, C_val):
        if self.entity._type == CM_Type.TP:
            return self.multiplier * C_val.tp
        elif self.entity._type == CM_Type.FP:
            return self.multiplier * C_val.fp
        elif self.entity._type == CM_Type.FN:
            return self.multiplier * C_val.fn
        elif self.entity._type == CM_Type.TN:
            return self.multiplier * C_val.tn
        elif self.entity._type == CM_Type.AP:
            return self.multiplier * C_val.ap
        elif self.entity._type == CM_Type.AN:
            return self.multiplier * C_val.an
        elif self.entity._type == CM_Type.PP:
            return self.multiplier * C_val.pp
        elif self.entity._type == CM_Type.PN:
            return self.multiplier * C_val.pn
        elif self.entity._type == CM_Type.ALL:
            return self.multiplier * C_val.all

    def compute_scaling(self, m, info):
        n = m-1
        ks = np.linspace(0, m-1, m)
        res = ConstantOverPQ()

        if self.entity._type == CM_Type.TP:
            res.cPQ = self.multiplier * np.ones((m,m))
        elif self.entity._type == CM_Type.FP:
            res.cPQ0 = self.multiplier * np.ones((m,m))
        elif self.entity._type == CM_Type.FN:
            res.cP0Q = self.multiplier * np.ones((m,m))
        elif self.entity._type == CM_Type.TN:
            res.cP0Q0 = self.multiplier * np.ones((m,m))
        elif self.entity._type == CM_Type.AP:
            res.c = self.multiplier * np.tile(ks, (m,1))
        elif self.entity._type == CM_Type.AN:
            res.c = self.multiplier * np.tile(n - ks, (m,1))
        elif self.entity._type == CM_Type.PP:
            res.c = self.multiplier * np.tile(ks, (m,1)).T
        elif self.entity._type == CM_Type.PN:
            res.c = self.multiplier * np.tile(n - ks, (m,1)).T
        elif self.entity._type == CM_Type.ALL:
            res.c = self.multiplier * n * np.ones((m,m))

        return res


class EXPR_UnaryIdentity(CM_Expression):
    def __init__(self, multiplier = 1):
        self.multiplier = multiplier

        super().__init__()
        self.is_constant = True

    def __pos__(self):
        return EXPR_UnaryIdentity(self.multiplier)

    def __neg__(self):
        return EXPR_UnaryIdentity(-self.multiplier)
    
    def __mul__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryIdentity(self.multiplier * other)
        else:
            # TODO: implement 
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __repr__(self):
        return str(self.multiplier) 

    def compute_value(self, C_val):
        return self.multiplier

    def compute_scaling(self, m, info):
        res = ConstantOverPQ()
        res.c = self.multiplier * np.ones((m,m))
        return res


class EXPR_UnaryExpr(CM_Expression):
    def __init__(self, expr, multiplier = 1):
        self.expr = expr 
        self.multiplier = multiplier

        super().__init__()
        self.is_linear_tp_tn = expr.is_linear_tp_tn
        self.depends_cell_cm = expr.depends_cell_cm
        self.depends_actual_sum = expr.depends_actual_sum
        self.depends_predicted_sum = expr.depends_predicted_sum
        self.is_constant = expr.is_constant
        self.needs_adv_sum_marg = expr.needs_adv_sum_marg
        self.needs_pred_sum_marg = expr.needs_pred_sum_marg
        self.is_constraint = expr.is_constraint

    def __mul__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryExpr(self.expr, self.multiplier * other)
        elif isinstance(other, CM_Expression):
            return EXPR_Multiplication(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Multiplication(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" * operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")

    def __truediv__(self, other):
        if (type(other) == int or type(other) == float):
            return EXPR_UnaryExpr(self.expr, self.multiplier / other)
        elif isinstance(other, CM_Expression):
            return EXPR_Fraction(self, other)
        elif isinstance(other, CM_Entity):
            return EXPR_Fraction(self, EXPR_UnaryEntity(other))
        else:
            raise OperationUndefined(" / operation over " + str(type(self)) +
                    " and " + str(type(other)) + "is undefined")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __repr__(self):
        if self.multiplier == 1:
            return str(self.expr)
        elif self.multiplier == -1:
            return "-" + "(" + str(self.expr) + ")"
        else:
            return str(self.multiplier) + " " + "(" + str(self.expr) + ")"

    def compute_value(self, C_val):
        return self.multiplier * self.expr.compute_value(C_val)

    def compute_scaling(self, m, info):
        return self.multiplier * self.expr.compute_scaling(m, info)


class EXPR_Fraction(CM_Expression):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator 

        super().__init__()
        if denominator.is_constant:
            self.is_linear_tp_tn = numerator.is_linear_tp_tn
            self.depends_cell_cm = numerator.depends_cell_cm
            self.depends_actual_sum = numerator.depends_actual_sum
            self.depends_predicted_sum = numerator.depends_predicted_sum
            self.is_constant = numerator.is_constant
        else:
            if denominator.depends_cell_cm:
                self.is_linear_tp_tn = False
            else:
                self.is_linear_tp_tn = numerator.is_linear_tp_tn
            self.depends_cell_cm = numerator.depends_cell_cm or denominator.depends_cell_cm
            self.depends_actual_sum = numerator.depends_actual_sum or denominator.depends_actual_sum
            self.depends_predicted_sum = numerator.depends_predicted_sum or denominator.depends_predicted_sum
            self.is_constant = False

        self.needs_adv_sum_marg = True if denominator.depends_actual_sum else numerator.needs_adv_sum_marg
        self.needs_pred_sum_marg = True if denominator.depends_predicted_sum else numerator.needs_pred_sum_marg     
        self.is_constraint = numerator.is_constraint or denominator.is_constraint

    def __repr__(self):
        return "(" + str(self.numerator) + ")" + " / " + "(" + str(self.denominator) + ")"

    def compute_value(self, C_val):
        return self.numerator.compute_value(C_val) / self.denominator.compute_value(C_val)

    def compute_scaling(self, m, info):
        res_num = self.numerator.compute_scaling(m, info)
        res_den = self.denominator.compute_scaling(m, info)

        if res_den.is_constant():
            C = res_den.c
            invC = np.zeros((m,m))
            if info.special_case_positive and info.special_case_negative:
                invC[1:m-1,1:m-1] = 1 / C[1:m-1,1:m-1]
            elif info.special_case_positive:
                invC[1:,1:] = 1 / C[1:,1:]
            elif info.special_case_negative:
                invC[:m-1,:m-1] = 1 / C[:m-1,:m-1]
            else:
                invC = 1 / C
            return res_num * invC
        else:
            raise UnsupportedMetric("The metric contains unsupported operations")


class EXPR_Multiplication(CM_Expression):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2 

        super().__init__()
        if not expr1.depends_cell_cm:
            self.is_linear_tp_tn = expr2.is_linear_tp_tn
        elif not expr2.depends_cell_cm:
            self.is_linear_tp_tn = expr1.is_linear_tp_tn
        else:
            self.is_linear_tp_tn = False
        self.depends_cell_cm = expr1.depends_cell_cm or expr2.depends_cell_cm
        self.depends_actual_sum = expr1.depends_actual_sum or expr2.depends_actual_sum
        self.depends_predicted_sum = expr1.depends_predicted_sum or expr2.depends_predicted_sum
        self.is_constant = expr1.is_constant and expr2.is_constant

        if expr1.depends_actual_sum:
            if expr2.is_constant:
                self.needs_adv_sum_marg = expr1.needs_adv_sum_marg
            else:
                self.needs_adv_sum_marg = True
        elif expr2.depends_actual_sum:
            if expr1.is_constant:
                self.needs_adv_sum_marg = expr2.needs_adv_sum_marg
            else:
                self.needs_adv_sum_marg = True
        else:
            self.needs_adv_sum_marg = False

        if expr1.depends_predicted_sum:
            if expr2.is_constant:
                self.needs_pred_sum_marg = expr1.needs_pred_sum_marg
            else:
                self.needs_pred_sum_marg = True
        elif expr2.depends_predicted_sum:
            if expr1.is_constant:
                self.needs_pred_sum_marg = expr2.needs_pred_sum_marg
            else:
                self.needs_pred_sum_marg = True
        else:
            self.needs_pred_sum_marg = False

        self.is_constraint = expr1.is_constraint or expr2.is_constraint
        
    def __repr__(self):
        return "(" + str(self.expr1) + ")" + " * " + "(" + str(self.expr2) + ")"

    def compute_value(self, C_val):
        return self.expr1.compute_value(C_val) * self.expr2.compute_value(C_val)

    def compute_scaling(self, m, info):
        res_ex1 = self.expr1.compute_scaling(m, info)
        res_ex2 = self.expr2.compute_scaling(m, info)

        if res_ex1.is_constant():
            return res_ex2 * res_ex1.c
        elif res_ex2.is_constant():
            return res_ex1 * res_ex2.c
        else:
            raise UnsupportedMetric("The metric contains unsupported operations")


class EXPR_Power(CM_Expression):
    def __init__(self, expr, power):
        self.expr = expr
        self.power = power

        super().__init__()
        if expr.depends_cell_cm:
            self.is_linear_tp_tn = False
        else:
            self.is_linear_tp_tn = expr.is_linear_tp_tn
        self.depends_cell_cm = expr.depends_cell_cm
        self.depends_actual_sum = expr.depends_actual_sum
        self.depends_predicted_sum = expr.depends_predicted_sum
        self.is_constant = expr.is_constant
        self.needs_adv_sum_marg = True if expr.depends_actual_sum else expr.needs_adv_sum_marg
        self.needs_pred_sum_marg = True if expr.depends_predicted_sum else expr.needs_pred_sum_marg     
        self.is_constraint = expr.is_constraint

    def __repr__(self):
        return "(" + str(self.expr) + ")**" + str(self.power) 

    def compute_value(self, C_val):
        return self.expr.compute_value(C_val) ** self.power

    def compute_scaling(self, m, info):
        res_expr = self.expr.compute_scaling(m, info)
        if res_expr.is_constant():
            res_expr.c = res_expr.c ** self.power
            return res_expr
        else:
            raise UnsupportedMetric("The metric contains unsupported operations")

        
class EXPR_FunctionCall(CM_Expression):
    def __init__(self, function, expr):
        self.function = function
        self.expr = expr 

        super().__init__()
        if expr.depends_cell_cm:
            self.is_linear_tp_tn = False
        else:
            self.is_linear_tp_tn = expr.is_linear_tp_tn
        self.depends_cell_cm = expr.depends_cell_cm
        self.depends_actual_sum = expr.depends_actual_sum
        self.depends_predicted_sum = expr.depends_predicted_sum
        self.is_constant = expr.is_constant
        self.needs_adv_sum_marg = True if expr.depends_actual_sum else expr.needs_adv_sum_marg
        self.needs_pred_sum_marg = True if expr.depends_predicted_sum else expr.needs_pred_sum_marg  
        self.is_constraint = expr.is_constraint
    
    def __repr__(self):
        return str(self.function) + "(" + str(self.expr) + ")"

    def compute_value(self, C_val):
        fn = getattr(math, self.function)
        return fn( self.expr.compute_value(C_val) )

    def compute_scaling(self, m, info):
        res_expr = self.expr.compute_scaling(m, info)
        if res_expr.is_constant():
            fn = getattr(math, self.function)
            res_expr.c = np.vectorize(fn)(res_expr.c)
            return res_expr
        else:
            raise UnsupportedMetric("The metric contains unsupported operations")


class EXPR_Addition(CM_Expression):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2 

        super().__init__()
        self.is_linear_tp_tn = expr1.is_linear_tp_tn and expr2.is_linear_tp_tn
        self.depends_cell_cm = expr1.depends_cell_cm or expr2.depends_cell_cm
        self.depends_actual_sum = expr1.depends_actual_sum or expr2.depends_actual_sum
        self.depends_predicted_sum = expr1.depends_predicted_sum or expr2.depends_predicted_sum
        self.is_constant = expr1.is_constant and expr2.is_constant
        self.needs_adv_sum_marg = expr1.needs_adv_sum_marg or expr2.needs_adv_sum_marg
        self.needs_pred_sum_marg = expr1.needs_pred_sum_marg or expr2.needs_pred_sum_marg
        self.is_constraint = expr1.is_constraint or expr2.is_constraint
        
    def __repr__(self):
        return "(" + str(self.expr1) + ")" + " + " + "(" + str(self.expr2) + ")"

    def compute_value(self, C_val):
        return self.expr1.compute_value(C_val) + self.expr2.compute_value(C_val)

    def compute_scaling(self, m, info):
        res_ex1 = self.expr1.compute_scaling(m, info)
        res_ex2 = self.expr2.compute_scaling(m, info)

        return res_ex1 + res_ex2


class EXPR_Constraint(CM_Expression):
    def __init__(self, expr, threshold):
        self.expr = expr
        self.threshold = threshold

        super().__init__()
        self.is_linear_tp_tn = False
        self.depends_cell_cm = expr.depends_cell_cm
        self.depends_actual_sum = expr.depends_actual_sum
        self.depends_predicted_sum = expr.depends_predicted_sum
        self.is_constant = expr.is_constant
        self.needs_adv_sum_marg = True if expr.depends_actual_sum else expr.needs_adv_sum_marg
        self.needs_pred_sum_marg = True if expr.depends_predicted_sum else expr.needs_pred_sum_marg    
        self.is_constraint = True

    def __repr__(self):
        return "(" + str(self.expr) + ") >= " + str(self.threshold) 

    def compute_value(self, C_val):
        return self.expr.compute_value(C_val) >= self.threshold

    def compute_scaling(self, m, info):
        raise OperationUndefined("Computing scaling over a constraint is not supported") 


## functions
def sqrt(x):
    if isinstance(x, CM_Expression):
        return EXPR_FunctionCall("sqrt", x)
    elif isinstance(x, CM_Entity):
        return EXPR_FunctionCall("sqrt", EXPR_UnaryEntity(x))
    elif (type(x) == int or type(x) == float):
        return math.sqrt(x)
    else:
        raise OperationUndefined(" sqrt operation over " + str(type(x)) + "is undefined")
