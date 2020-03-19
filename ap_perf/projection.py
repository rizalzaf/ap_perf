import numpy as np
from scipy.optimize import minimize
from numba import jit

stored_init_rho = [0.5, 0.5]
stored_init_lambda = [np.array([0.5,]), np.array([0.5,])]

@jit(nopython=True)
def solve_p_given_abk(a, b, k):
    n = len(a)

    raw = a - b
    sorted_raw = np.sort(raw)[::-1]

    neg_cut_id = n - np.searchsorted(sorted_raw[::-1], 0.0)
    if neg_cut_id == n:
        sraw_neg = np.zeros(0)
        n_neg = 0
    else:
        sraw_neg = sorted_raw[neg_cut_id:]
        n_neg = len(sraw_neg)
    
    sum_p = np.sum(sorted_raw[0:neg_cut_id])
    if sorted_raw[0] <=  sum_p / k:
        return np.maximum(raw, 0)
    
    # initialize 
    i = 1                   # #x in sum_p_edge
    j = neg_cut_id - 1      # #x in sum_p_else
    l = 0                   # #x in sum_p_neg

    sum_p_edge = sorted_raw[0]
    sum_p_else = sum_p - sorted_raw[0]
    sum_p_neg = 0.0

    # solution
    sol_r = 0.0
    p_add = 0.0

    while True: 

        next_r = (sum_p_else + sum_p_neg + sum_p_edge * (j + l) / (k - i)) / (k - i + (i * (j + l)) / (k - i))
        p_else_add = (sum_p_edge - i * next_r) / (k - i)

        if l < n_neg and p_else_add + sraw_neg[l] > 0 :
            sum_p_neg += sraw_neg[l]
            l += 1
            continue
        
        if  i == k-1 or next_r + 1e-6 >= sorted_raw[i] + p_else_add :     # 1e-6 for floating point error
            sol_r = next_r
            p_add = p_else_add

            break
        
        sum_p_edge += sorted_raw[i]
        sum_p_else -= sorted_raw[i]
        i += 1
        j -= 1

    res = np.minimum(np.maximum(raw + p_add, 0), sol_r)
    return res


@jit(nopython=True)
def obj_rho(rho_arr, A):
    rho = rho_arr[0]
    n = A.shape[0]

    # to store best P
    best_P = np.zeros((n,n))

    obj = 0.0
    grad = np.array([0.0])

    for i in range(n):
        k = i + 1
        a = A[:,i]
        b = np.ones(n) * rho / k

        opt_p = solve_p_given_abk(a, b, k)
        best_P[:,i] = opt_p

        obj += np.sum(opt_p) * rho / k + np.linalg.norm(opt_p - a) ** 2 / 2
        grad += np.sum(opt_p) / k

    obj -= rho
    grad -= 1

    return -obj, -grad, best_P


def marginal_projection(A, init_storage_id = 0):    
    global stored_init_rho

    n = A.shape[0]

    f_rho = lambda rho: obj_rho(rho, A)[0]
    g_rho = lambda rho: obj_rho(rho, A)[1]

    fg_rho = lambda rho: obj_rho(rho, A)[0:2]

    init_rho = [ stored_init_rho[init_storage_id] ]
    res_rho = minimize(fg_rho, init_rho, method='L-BFGS-B', jac=True, bounds=((0.0, np.inf),), options={'maxiter' : 20})

    opt_P = obj_rho(res_rho.x, A)[2]
    stored_init_rho[init_storage_id] = res_rho.x[0]

    # # avoid greater than one of sum P that may result from floating point error
    # max_pi = np.max(np.sum(opt_P, axis=1))
    # if max_pi > 1.0:
    #     opt_P -= max_pi / n
    
    # avoid non negativity that may result from floating point error
    opt_P[opt_P <= 0.0] = 0.0

    return opt_P

# projecttion with constraints
# @jit(nopython=True)
def obj_rho_lambda(rho_lambda, A, B_list, c_list, tau_list):
    rho = rho_lambda[0]
    lda = rho_lambda[1:]

    n = A.shape[0]
    ncs = len(B_list)

    B_wsum = np.zeros((n,n))
    for j in range(ncs):
        B_wsum += lda[j] * B_list[j]

    # to store best P
    best_P = np.zeros((n,n))

    obj = 0.0
    grad = np.zeros(len(rho_lambda))

    for i in range(n):
        k = i + 1
        a = A[:,i]
        b = np.ones(n) * rho / k - B_wsum[:,i]

        opt_p = solve_p_given_abk(a, b, k)
        # opt_p = solve_p_given_abk_guess(a, b, k, int(0.6 *k))
        best_P[:,i] = opt_p

        obj += np.dot(opt_p, b) + np.linalg.norm(opt_p - a) ** 2 / 2

    obj -= rho
    for j in range(ncs):
        obj += lda[j] * (tau_list[j] - c_list[j])

    # compute grad
    ks = np.linspace(1, n, n)
    grad[0] += np.sum(best_P / ks)
    for j in range(ncs):
        grad[j+1] += (-np.sum(B_list[j] * best_P))
 
    grad[0] -= 1.0
    for j in range(ncs):
        grad[j+1] += (tau_list[j] - c_list[j])

    return -obj, -grad, best_P


def marginal_projection_with_constraint(A, B_list, c_list, tau_list, init_storage_id = 0):    
    global stored_init_rho
    global stored_init_lambda

    # initialize
    ncs = len(B_list)

    # pull stored parameters
    init_rho = stored_init_rho[init_storage_id]
    if len(stored_init_lambda[init_storage_id]) != ncs :
        stored_init_lambda[init_storage_id] = np.ones(ncs) * 0.5
    init_lambda = stored_init_lambda[init_storage_id]

    n = A.shape[0]

    f_rho = lambda rholambda: obj_rho_lambda(rholambda, A, B_list, c_list, tau_list)[0]
    g_rho = lambda rholambda: obj_rho_lambda(rholambda, A, B_list, c_list, tau_list)[1]

    x0 = np.hstack((init_rho, init_lambda))
    bnd = ((0.0, None),) * (ncs + 1)
    res_rho = minimize(f_rho, x0, method='L-BFGS-B', jac=g_rho, bounds=bnd, options={'maxiter' : 20})

    sol = res_rho.x
    opt_P = obj_rho_lambda(sol, A, B_list, c_list, tau_list)[2]

    # store pars
    stored_init_rho[init_storage_id] = sol[0:1]
    stored_init_lambda[init_storage_id] = sol[1:]

    # # avoid greater than one of sum P that may result from floating point error
    # max_pi = np.max(np.sum(opt_P, axis=1))
    # if max_pi > 1.0:
    #     opt_P -= max_pi / n
    # avoid non negativity that may result from floating point error
    opt_P[opt_P <= 0.0] = 0.0

    return opt_P


## Proximal functions
def prox_max_sumlargest(A, rho = 1.0, init_storage_id = 0):
    return A - marginal_projection(A * rho, init_storage_id) / rho

def prox_max_sumlargest_with_constraint(A, B_list, c_list, tau_list, rho = 1.0, init_storage_id = 0):  
    return A - marginal_projection_with_constraint(A * rho, B_list, c_list, tau_list, init_storage_id) / rho


## max sum k largest, non neg
# compute value
@jit(nopython=True)
def sum_k_largest(a, k):
    sa = np.sort(a)[::-1]
    return np.sum(sa[:k])

@jit(nopython=True)
def sumlargest(A):
    n = A.shape[0]
    sl = np.zeros(n)
    for i in range(n):
        sl[i] = sum_k_largest(A[:,i], i)
    return sl

@jit(nopython=True)
def max_sumlargest(A, non_neg = True):
    sl = sumlargest(A)
    m = np.max(sl)
    if non_neg:
        m = max(0, m)
    return m


@jit(nopython=True)
def find_opt_p(B, mode="max"):
    n = B.shape[0]

    SB = np.zeros((n,n), np.int64)
    for i in range(n):
        SB[:,i] = np.argsort(B[:,i]) if mode == "min" else np.argsort(-B[:,i])

    best_i = -1
    best_sum = 0.0
    for i in range(n):
        sum_i = np.sum(B[SB[0:i+1,i],i])
        if mode == "max" and sum_i > best_sum:
            best_i = i
            best_sum = sum_i
        elif mode == "min" and sum_i < best_sum: 
            best_i = i
            best_sum = sum_i
        
    P = np.zeros((n,n))
    if best_i >= 0:
        P[SB[0:best_i+1, best_i], best_i] = 1.0

    return P # , best_sum


## reset storage
def reset_projection_storage():
    global stored_init_rho
    global stored_init_lambda

    stored_init_rho = [0.5, 0.5]
    stored_init_lambda = [np.array([0.5,]), np.array([0.5,])]