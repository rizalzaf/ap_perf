# from expression import CM_Entity, CM_Type, CM_Category, CM_Expression, \
#     UnsupportedMetric, InputMismatch, sqrt
# from projection import max_sumlargest, marginal_projection, \
#     prox_max_sumlargest, prox_max_sumlargest_with_constraint, reset_projection_storage
from ap_perf.expression import CM_Entity, CM_Type, CM_Category, CM_Expression, \
    UnsupportedMetric, InputMismatch, sqrt
from ap_perf.projection import max_sumlargest, marginal_projection, \
    prox_max_sumlargest, prox_max_sumlargest_with_constraint, reset_projection_storage
import numpy as np
import scipy as sp
import abc

class Confusion_Matrix:
    def __init__(self):
        self.tp = CM_Entity(CM_Type.TP, CM_Category.CELL) 
        self.fp = CM_Entity(CM_Type.FP, CM_Category.CELL) 
        self.fn = CM_Entity(CM_Type.FN, CM_Category.CELL) 
        self.tn = CM_Entity(CM_Type.TN, CM_Category.CELL) 
        self.ap = CM_Entity(CM_Type.AP, CM_Category.ACTUAL_SUM) 
        self.an = CM_Entity(CM_Type.AN, CM_Category.ACTUAL_SUM) 
        self.pp = CM_Entity(CM_Type.PP, CM_Category.PREDICTION_SUM)
        self.pn = CM_Entity(CM_Type.PN, CM_Category.PREDICTION_SUM)
        self.all = CM_Entity(CM_Type.ALL, CM_Category.ALL_SUM)

class CM_Value:
    def __init__(self, yhat, y):
        self.all = len(y)
        self.tp = np.dot(yhat, y)
        self.ap = np.sum(y == 1)
        self.pp = np.sum(yhat == 1)

        self.an = self.all - self.ap
        self.pn = self.all - self.pp

        self.fp = self.pp - self.tp
        self.fn = self.ap - self.tp
        self.tn = self.an - self.fp

# for storing infos
class Info:
    pass    

# for storing ADMM data
class OptData:
    pass


class PerformanceMetric:
    def __init__(self):
        pass

    @abc.abstractmethod
    def define(self, C: Confusion_Matrix):
        pass    # must be overloaded

    def constraint(self, C: Confusion_Matrix):
        return None     # the default is without constraint

    def enforce_special_case_positive(self):
        self.special_case_positive = True
    
    def enforce_special_case_negative(self):
        self.special_case_negative = True

    def set_cs_special_case_positive(self, sc_list):
        n_constraint = len(self.constraint_list)
        if isinstance(sc_list, bool):
            self.cs_special_case_positive_list = [sc_list] * n_constraint
        elif isinstance(sc_list, list):
            if len(sc_list) == n_constraint:
                self.cs_special_case_positive_list = sc_list
            else:
                raise InputMismatch("Input length must be the same as the number of constraints")
        else:
            raise InputMismatch("The input must be a boolean or list of booleans")
    
    def set_cs_special_case_negative(self, sc_list):
        n_constraint = len(self.constraint_list)
        if isinstance(sc_list, bool):
            self.cs_special_case_negative_list = [sc_list] * n_constraint
        elif isinstance(sc_list, list):
            if len(sc_list) == n_constraint:
                self.cs_special_case_negative_list = sc_list
            else:
                raise InputMismatch("Input length must be the same as the number of constraints")
        else:
            raise InputMismatch("The input must be a boolean or list of booleans")
        
    
    def initialize(self):
        self.C = Confusion_Matrix()

        self.metric_expr = self.define(self.C)
        self.valid = self.metric_expr.is_linear_tp_tn and self.metric_expr.depends_cell_cm

        self.needs_adv_sum_marg = self.metric_expr.needs_adv_sum_marg
        self.needs_pred_sum_marg = self.metric_expr.needs_pred_sum_marg

        # check flags for special cases
        if not hasattr(self, "special_case_positive"):
            self.special_case_positive = False
        if not hasattr(self, "special_case_negative"):
            self.special_case_negative = False

        if not self.valid:
            raise UnsupportedMetric("The metric and/or the constraints are not supported")

        # check constraints
        self.constraint_expr = self.constraint(self.C)
        if self.constraint_expr is None:
            self.constraint_list = []
        elif isinstance(self.constraint_expr, CM_Expression):
            self.constraint_list = [self.constraint_expr]
            if not self.constraint_expr.is_constraint:
                raise UnsupportedMetric("The constraint is unsupported")
        elif isinstance(self.constraint_expr, list):
            self.constraint_list = self.constraint_expr
            for ex in self.constraint_expr:
                if not ex.is_constraint:
                    raise UnsupportedMetric("The constraint is unsupported")
        else:
            raise UnsupportedMetric("The constraint is unsupported")

        # check flags for special cases for constraints
        n_constraint = len(self.constraint_list)
        if not hasattr(self, "cs_special_case_positive_list"):
            self.cs_special_case_positive_list = [False] * n_constraint
        if not hasattr(self, "cs_special_case_negative_list"):
            self.cs_special_case_negative_list = [False] * n_constraint
    
    def compute_metric(self, yhat, y):
        # check for special cases
        if self.special_case_positive:
            if np.sum(np.equal(y, 0)) == len(y) and np.sum(np.equal(yhat, 0)) == len(yhat):
                return 1.0
            elif np.sum(np.equal(y, 0)) == len(y):
                return 0.0
            elif np.sum(np.equal(yhat, 0)) == len(yhat):
                return 0.0
        
        if self.special_case_negative:
            if np.sum(np.equal(y, 1)) == len(y) and np.sum(np.equal(yhat, 1)) == len(yhat):
                return 1.0
            elif np.sum(np.equal(y, 1)) == len(y):
                return 0.0
            elif np.sum(np.equal(yhat, 1)) == len(yhat):
                return 0.0
    
        C_val = CM_Value(yhat, y)
        val = self.metric_expr.compute_value(C_val)
        return val


    def compute_constraints(self, yhat, y):
        nconst = len(self.constraint_list)
        vals = np.zeros(nconst)

        C_val = CM_Value(yhat, y)

        for ics in range(nconst):
            # check for special cases
            if self.cs_special_case_positive_list[ics]:
                if np.sum(np.equal(y, 0)) == len(y) and np.sum(np.equal(yhat, 0)) == len(yhat):
                    vals[ics] = 1.0
                    continue
                elif np.sum(np.equal(y, 0)) == len(y):
                    vals[ics] = 0.0
                    continue
                elif np.sum(np.equal(yhat, 0)) == len(yhat):
                    vals[ics] = 0.0
                    continue
            
            if self.cs_special_case_negative_list[ics]:
                if np.sum(np.equal(y, 1)) == len(y) and np.sum(np.equal(yhat, 1)) == len(yhat):
                    vals[ics] = 1.0
                    continue
                elif np.sum(np.equal(y, 1)) == len(y):
                    vals[ics] = 0.0
                    continue
                elif np.sum(np.equal(yhat, 1)) == len(yhat):
                    vals[ics] = 0.0
                    continue
            
            cs_expr = self.constraint_list[ics]
            lhs_expr = cs_expr.expr     # left hand side of the expr >= tau
        
            vals[ics] = lhs_expr.compute_value(C_val)
        
        return vals


    def __compute_multipliers(self, n):
        # compute multipliers
        m = n + 1

        # for passing infos
        info = Info()
        info.needs_pred_sum_marg = self.needs_pred_sum_marg
        info.needs_adv_sum_marg = self.needs_adv_sum_marg
        info.special_case_positive = self.special_case_positive
        info.special_case_negative = self.special_case_negative

        # compute grads
        self.multiplier_pq = self.metric_expr.compute_scaling(m, info)  


    def __generate_constraints_on_p(self, y):
        n = len(y)

        self.B_list = []
        self.c_list = []
        self.tau_list = []

        k = int(np.sum(y == 1))
        Q = np.zeros((n, n))
        if k > 0:
            Q[:, k-1] = y

        for ics in range(len(self.constraint_list)):
            cs_expr = self.constraint_list[ics]
            lhs_expr = cs_expr.expr     # left hand side of the expr >= tau
            tau = cs_expr.threshold     # right hand side of the expr >= tau

            info = Info()
            info.needs_pred_sum_marg = lhs_expr.needs_pred_sum_marg
            info.needs_adv_sum_marg = lhs_expr.needs_adv_sum_marg
            info.special_case_positive = self.cs_special_case_positive_list[ics]
            info.special_case_negative = self.cs_special_case_negative_list[ics]

            m = n + 1

            # compute scaling
            multiplier_pq = lhs_expr.compute_scaling(m, info)  

            # compute grad p
            dP, const_p = PerformanceMetric.__compute_grad_np(Q, multiplier_pq, info)

            # store to list
            self.B_list.append(dP)
            self.c_list.append(const_p)
            self.tau_list.append(tau)


    @staticmethod
    def __compute_grad_np(Q, multiplier_pq, info):
        
        # compute objective
        n = Q.shape[0]
        
        # compute P(zerovec) Q(zerovec)
        ks = np.linspace(1, n, n)
        iks = 1 / ks
        KS = np.tile(ks, (n,1))
        IKS = np.tile(iks, (n,1))
        
        # compute Q(zerovec)
        qsum_zero = 1 - np.sum(np.multiply(Q, IKS))
        q_zerovec = np.multiply(qsum_zero, np.ones((n, 1)))
        
        # compute P0 and Q0
        qsum = np.sum(np.multiply(Q, IKS), axis=0) 
        Q0 = np.matmul(np.ones((n,n)), np.multiply(Q, IKS)) - Q

        Q_0k = np.hstack((np.zeros((n,1)), Q))
        Q0_0k = np.hstack((q_zerovec, Q0))
        qsum_0k = np.hstack((qsum_zero, qsum))

        # regular idx
        if info.special_case_negative and info.special_case_positive:
            idx = range(1,n)
        elif info.special_case_positive:
            idx = range(1,n+1)
        elif info.special_case_negative:
            idx = range(0,n)
        else:
            idx = range(0,n+1)
        idn = range(0,n)

        dP_idx_defined = False
        # collect for regular idx
        if multiplier_pq.cPQ is not None: 
            dP_idx = np.matmul(Q_0k[np.ix_(idn, idx)], multiplier_pq.cPQ[np.ix_(idx, idx)].T)
            dP_idx_defined = True
        if multiplier_pq.cPQ0 is not None:
            if dP_idx_defined:
                dP_idx += np.matmul(Q0_0k[np.ix_(idn, idx)], multiplier_pq.cPQ0[np.ix_(idx, idx)].T)
            else:
                dP_idx = np.matmul(Q0_0k[np.ix_(idn, idx)], multiplier_pq.cPQ0[np.ix_(idx, idx)].T)
                dP_idx_defined = True

        dP0_idx_defined = False
        if multiplier_pq.cP0Q is not None: 
            dP0_idx = np.matmul(Q_0k[np.ix_(idn, idx)], multiplier_pq.cP0Q[np.ix_(idx, idx)].T)
            dP0_idx_defined = True
        if multiplier_pq.cP0Q0 is not None: 
            if dP0_idx_defined:
                dP0_idx += np.matmul(Q0_0k[np.ix_(idn, idx)], multiplier_pq.cP0Q0[np.ix_(idx, idx)].T)
            else:
                dP0_idx = np.matmul(Q0_0k[np.ix_(idn, idx)], multiplier_pq.cP0Q0[np.ix_(idx, idx)].T)
                dP0_idx_defined = True

        dpsum_idx_defined = False
        if multiplier_pq.c is not None:
            dpsum_idx = multiplier_pq.c[np.ix_(idx, idx)] @ qsum_0k[np.ix_(idx)]
            dpsum_idx_defined = True

        # init dP
        dP = np.zeros((n,n))
        dp_zerovec = np.zeros((n,1))
        dpsum = np.zeros(n)

        # put it in dP
        if info.special_case_negative and info.special_case_positive:
            dp_zerovec = q_zerovec / n
            dp_onevec = (Q[:,n-1] / n).reshape((n,1))
            if dP_idx_defined: 
                dP = np.hstack([dP_idx[:,0:n-1], dp_onevec])
            if dP0_idx_defined: 
                dP0 = np.hstack([dP0_idx[:,0:n-1], np.zeros((n,1))])
            if dpsum_idx_defined:
                dpsum = np.hstack([dpsum_idx[0:n-1], 0.0])
        elif info.special_case_positive:
            if dP_idx_defined: 
                dP = dP_idx
            if dP0_idx_defined: 
                dP0 = dP0_idx
            if dpsum_idx_defined:
                dpsum = dpsum_idx
            dp_zerovec = q_zerovec / n
        elif info.special_case_negative:
            if dP_idx_defined: 
                dp_onevec = (Q[:,n-1] / n).reshape((n,1))
                dP = np.hstack([dP_idx[:,1:n], dp_onevec])
            if dP0_idx_defined: 
                dP0 = np.hstack([dP0_idx[:,1:n], np.zeros((n,1))])
                dp_zerovec = dP0_idx[:,0]
            if dpsum_idx_defined:
                dpsum = np.hstack([dpsum_idx[1:n], 0.0])
        else:
            if dP_idx_defined: 
                dP = dP_idx[:,1:n+1]
            if dP0_idx_defined: 
                dP0 = dP0_idx[:,1:n+1]
                dp_zerovec = dP0_idx[:,0]
            if dpsum_idx_defined:
                dpsum = dpsum_idx[1:n+1]

        # zerovec cases if no special cases enforced for positive
        if not info.special_case_positive:
            if dpsum_idx_defined:
                dp_zerovec += (dpsum_idx[0] / n) * np.ones((n, 1))

        ## transform dP0 and dQ0 as dP and dQ
        if dP0_idx_defined: 
            dP += np.matmul(np.ones((n,n)), np.multiply(dP0, IKS)) - dP0

        # transform dpsum
        if dpsum_idx_defined:
            dP += np.multiply(np.matmul(np.ones((n,1)), np.reshape(dpsum, (1,n))), IKS)

        ## transform dp_zerovec and dq_zerovec as dP and dQ
        dP += (- np.multiply(np.sum(dp_zerovec), IKS))

        # constant
        const_p = 0.0
        if info.special_case_positive:
            const_p = np.sum(dp_zerovec)
        
        return dP, const_p


    #### ADMM based 
    def __compute_admm_matrices(self, n):
        
        # for passing infos
        info = Info()
        info.needs_pred_sum_marg = self.needs_pred_sum_marg
        info.needs_adv_sum_marg = self.needs_adv_sum_marg
        info.special_case_positive = self.special_case_positive
        info.special_case_negative = self.special_case_negative

        # compute grads
        multiplier_pq = self.multiplier_pq 

        ks = np.linspace(1, n, n)
        IK = np.diag(1 / ks)

        idx = range(1,n+1)
        idn = range(0,n)

        ## <P,  A Q B + Q C + D> + <Q, E>
        ## A = ones(n,n)

        A = np.ones((n,n))
        B = np.zeros((n,n))
        C = np.zeros((n,n))
        D = np.zeros((n,n))
        E = np.zeros((n,n))

        mult_u0v0 = 0.0

        # special case negative modify M
        if info.special_case_negative:
            if multiplier_pq.cPQ is not None: 
                multiplier_pq.cPQ[n, :] = 0.0
                multiplier_pq.cPQ[:, n] = 0.0
                multiplier_pq.cPQ[n, n] = 1.0
            if multiplier_pq.c is not None: 
                multiplier_pq.c[n, :] = 0.0
                multiplier_pq.c[:, n] = 0.0
                multiplier_pq.c[n, n] = 1.0

        # compute matrices 
        if multiplier_pq.cPQ is not None:
            C += multiplier_pq.cPQ[np.ix_(idx, idx)].T
        if multiplier_pq.cPQ0 is not None: 
            C -= multiplier_pq.cPQ0[np.ix_(idx, idx)].T
            B += IK @ multiplier_pq.cPQ0[np.ix_(idx, idx)].T
        
        if multiplier_pq.cP0Q is not None: 
            C -= multiplier_pq.cP0Q[np.ix_(idx, idx)].T
            B += multiplier_pq.cPQ0[np.ix_(idx, idx)].T @ IK
        if multiplier_pq.cP0Q0 is not None: 
            MP0Q0 = multiplier_pq.cP0Q0
            C += MP0Q0[np.ix_(idx, idx)].T
            B += n * IK @ MP0Q0[np.ix_(idx, idx)].T @ IK  -  MP0Q0[np.ix_(idx, idx)].T @ IK  -  IK @ MP0Q0[np.ix_(idx, idx)].T

            if not info.special_case_positive:
                B += MP0Q0[0, idx].reshape((n,1)) @ np.ones((1,n)) @ IK  -  n * IK @ MP0Q0[0, idx].reshape((n,1)) @ np.ones((1,n)) @ IK
                B += IK @ np.ones((n,1)) @ MP0Q0[idx, 0].reshape((1,n))  -  n * IK @ np.ones((n,1)) @ MP0Q0[idx, 0].reshape((1,n)) @ IK

                D += n * np.ones((n,1)) @ MP0Q0[idx, 0].reshape((1,n)) @ IK  -  np.ones((n,1)) @ MP0Q0[idx, 0].reshape((1,n))
                E += n * np.ones((n,1)) @ MP0Q0[0, idx].reshape((1,n)) @ IK  -  np.ones((n,1)) @ MP0Q0[0, idx].reshape((1,n))

                mult_u0v0 += MP0Q0[0,0]
            
        if multiplier_pq.c is not None: 
            B += IK @ multiplier_pq.c[np.ix_(idx, idx)].T @ IK
            
            if not info.special_case_positive:
                B += IK @ np.ones((n,1)) @ multiplier_pq.c[idx, 0].reshape((1,n)) @ IK  -  IK @ multiplier_pq.c[0, idx].reshape((n,1)) * np.ones((1,n)) @ IK
                D += np.ones((n,1)) @ multiplier_pq.c[idx, 0].reshape((1,n)) @ IK
                E += np.ones((n,1)) @ multiplier_pq.c[0, idx].reshape((1,n)) @ IK

                mult_u0v0 += multiplier_pq.c[0, 0]
        
        if info.special_case_positive:
            B += IK @ np.ones((n,n)) @ IK
            D -= np.ones((n,n)) @ IK
            E -= np.ones((n,n)) @ IK
            ct = 1.0    
        else:
            B += IK @ np.ones((n,n)) @ IK * mult_u0v0
            D -= np.ones((n,n)) @ IK * mult_u0v0
            E -= np.ones((n,n)) @ IK * mult_u0v0
            ct = mult_u0v0    
        

        # eigen decomposition
        # precompute Matrices
        BC = n * B @ B.T + C @ B.T + B @ C.T
        CIinv = np.linalg.inv(C @ C.T + np.eye(n))
        BC_Cinv = BC @ CIinv


        # find eigen decomposition of BC * CIinv using CIinv^{0.5} * BC * CIinv^{0.5}
        # CIinv^{0.5} * BC * CIinv^{0.5} is symmetric, so we get a nicer eigendecomposition
        # CIinv^{0.5} * BC * CIinv^{0.5} and BC * CIinv have the same eigen values
        sqC = sp.linalg.sqrtm(CIinv).real   # matrix sqrt: i.e  sqC * sqC = CIinv; always real since CIinv is posdef
        sqC_BC_sqC = sqC @ BC @ sqC     # it's symmetric


        sz, UZ = np.linalg.eigh(sqC_BC_sqC)     # eigh since it's symmetric
        sz = sz.real
        UZ = UZ.real

        ## convert to eigen decomposition over BC * CIinv, say:
        ## CIinv^{0.5} * BC * CIinv^{0.5} = U S U^-1, and
        ## BC * CIinv = V S V^-1, therefore:
        ## CIinv^{-0.5} CIinv^{0.5} * BC * CIinv^{0.5} CIinv^{0.5} = CIinv^{-0.5} U S U^-1 CIinv^{0.5}
        ## BC * CIinv = CIinv^{-0.5} U S U^-1 CIinv^{0.5}
        ## Hence: V = CIinv^{-0.5} U
        sbc = sz
        UBC = np.linalg.inv(sqC) @ UZ


        # eugen dec for A
        sa, UA = np.linalg.eigh(A)
        sa = sa.real
        UA = UA.real

        # sa * sbc' + 1
        sabc1 = (sa.reshape((n,1)) @ sbc.reshape((1,n)) + 1)

        # inverses
        UAinv = np.linalg.inv(UA)
        UBCinv = np.linalg.inv(UBC)

        # store matrices in opt_data
        opt_data = OptData()
        opt_data.n = n

        opt_data.A = A
        opt_data.B = B
        opt_data.C = C
        opt_data.D = D
        opt_data.E = E

        opt_data.ct = ct
        opt_data.CIinv = CIinv
        opt_data.BC_Cinv = BC_Cinv

        opt_data.UA = UA
        opt_data.UBC = UBC

        opt_data.UAinv = UAinv
        opt_data.UBCinv = UBCinv
        opt_data.sabc1 = sabc1

        self.opt_data = opt_data


    # use the stored matric calculation instead
    def __compute_grad_p(self, Q):
        # get stored matrices
        od = self.opt_data
        A = od.A; B = od.B; C = od.C; D = od.D; E = od.E; ct = od.ct 
        dP = A @ Q @ B + Q @ C + D
        const_p = np.sum(Q * E) + ct 

        return dP, const_p


    @staticmethod
    def __obj_admm_q(Q, A, B, C, D, E, ct):
        obj = max_sumlargest(A @ Q @ B + Q @ C + D) + np.sum(Q * E) + ct
        return obj


    def __solve_admm_q(self, PSI, y, **kwargs):
        n = PSI.shape[0]
        ks = np.linspace(1, n, n)

        if len(self.constraint_list) > 0:
            self.__generate_constraints_on_p(y)

        if (not hasattr(self, "multiplier_pq")) :
            self.__compute_multipliers(n)
        if (not hasattr(self, "opt_data")) :
            self.__compute_admm_matrices(n)
        if self.opt_data.n != n :
            self.__compute_multipliers(n)
            self.__compute_admm_matrices(n)


        # check options
        rho = 1.0
        max_iter = 100
        if 'rho' in kwargs:
            rho = kwargs['rho']
        if 'max_iter' in kwargs:
            max_iter = kwargs['max_iter']

        # prox function, check of there's constraints or not
        if len(self.constraint_list) == 0:
            # prox function
            prox_function = prox_max_sumlargest
        else:
            # prox function
            prox_function = lambda A, rho, init_storage_id: \
                prox_max_sumlargest_with_constraint(A, self.B_list, self.c_list, self.tau_list, rho, init_storage_id)


        # get stored matrices
        od = self.opt_data
        A = od.A; B = od.B; C = od.C; D = od.D; E = od.E; ct = od.ct 
        CIinv = od.CIinv; BC_Cinv = od.BC_Cinv
        UA = od.UA; UBC = od.UBC; UAinv = od.UAinv; UBCinv = od.UBCinv; sabc1 = od.sabc1

        EP = E - PSI

        Q = np.zeros((n,n))
        X = np.zeros((n,n))
        Z = np.zeros((n,n))

        U = np.zeros((n,n))
        W = np.zeros((n,n))

        # resetting the storage in the LBFGS for marginal projection
        reset_projection_storage()
        
        it = 1
        while True:

            # opt Q
            Q = marginal_projection( (rho * (X + W) - EP) / rho, 0 )  # init_storage_id = 0

            # opt Z
            Z = prox_function(A @ X @ B  +  X @ C + D + U, rho, 1 )   # init_storage_id = 1
 
            # opt X
            DZU = D - Z + U
            F = A @ DZU @ B.T + DZU @ C.T + W - Q

            UFU = UAinv @ (-F @ CIinv) @ UBC
            X1 = UFU / sabc1
            X = UA @ X1 @ UBCinv

            # opt dual
            U = U + A @ X @ B + X @ C + D - Z
            W = W + X - Q

            if it >= max_iter:
                break

            it += 1

        obj = PerformanceMetric.__obj_admm_q(Q, A, B, C, D, EP, ct)

        return Q, obj


    def objective(self, psi, y, **kwargs):

        n = len(psi)
        PSI = np.tile(psi, (n,1)).T

        Q, obj = self.__solve_admm_q(PSI, y, **kwargs)
        q = np.sum(Q, axis = 1)

        return obj, q