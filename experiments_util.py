from time import time
import numpy as np
from other_jd_algorithms import *
from rnojd import *


def random_error(AA, eps = 1e-5, norm_type = 'fro',pd = True):
    n = AA.shape[0]
    p = AA.shape[1]
    result = None
    nonpd = True
    while(nonpd):
        result = []
        for A in AA:
            E = np.random.normal(0,1,(p,p))
            E = (E + E.T) / 2
            E = eps * (1/(np.linalg.norm(E,norm_type) *np.sqrt(n))) * E
            result.append(((A + E) + (A + E).T) / 2.0)
        nonpd = False
        for A in result:
            try:
                np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                nonpd = True
        if not pd:
            break
    print("generation succeed")
    return np.array(result)

def random_jd_matrices(d = 5, n = 4, orthogonal = False):
    diagonals = 3*1e-2  +  np.abs(np.random.normal(size=(d, n)))
    V = np.random.randn(n, n)
    V =  V / np.linalg.norm(V,axis=0)
    if orthogonal:
        V,_ = np.linalg.qr(V)
    C = np.array([V.dot(d[:, None] * V.T) for d in diagonals])
    return C

def random_jd_matrices_negative(d = 5, n = 4, orthogonal = False):
    diagonals = np.random.normal(size=(d, n))
    V = np.random.randn(n, n)
    V =  V / np.linalg.norm(V,axis=0)
    if orthogonal:
        V,_ = np.linalg.qr(V)
    C = np.array([V.dot(d[:, None] * V.T) for d in diagonals])
    return C

def random_jd_ill_conditioned_matrices(d = 5, n = 4, orthogoal = False, max_power = 8):
    log_spaced_entries = np.logspace(0,max_power,n)
    diagonals = [np.random.permutation(log_spaced_entries) for _ in range(d)]
    V = np.random.randn(n, n)
    V =  V / np.linalg.norm(V,axis=0)
    if orthogoal:
        V,_ = np.linalg.qr(V)
    C = np.array([V.dot(d[:, None] * V.T) for d in diagonals])
    return C

def print_time_error(name, times,errors, bold = False):
    if not bold:
        output_str = name + " & " + "%.2f" + " & " + "$\\num{%.3g}$"  + " & " \
                    + "%.2f" + " & " + "$\\num{%.3g}$" + " & " \
                    + "%.2f" + " & " + "$\\num{%.3g}$" + "\\\\\n" 
    else:
        output_str = "{\\bf " + name + "}" + " & " + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$"  + " & " \
                    + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$" + " & " \
                    + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$" + "\\\\\n"
    print(output_str % (times[0], errors[0], times[1], errors[1], times[2], errors[2]))

def print_time_error_ill_conditioned(name, times,errors, bold = False):
    if not bold:
        output_str = name + " & " + "$\\num{%.2f}$" + " & " + "$\\num{%.3g}$"  + "\\\\\n" 
    print(output_str % (times, errors))

def offdiagonal_frobenius_square(A, by_column = False):
    """
    computes the frobenius norm of the off diagonal elements
    of the tensor A (k x m x m)
    Args:
        A: np.ndarray
            of shape k x m x m
    Returns:
        norm: np.ndarray
            the frobenius norm square of the offdiagonal of A
    """
    shape = A.shape
    identity_3d = np.zeros(shape)
    idx = np.arange(shape[1])
    identity_3d[:, idx, idx] = 1 
    mask = np.array(1 - identity_3d, dtype = np.int0)
    offdiag = A * mask
    if not by_column:
        loss = np.sum(np.power(offdiag,2))
        return loss
    else:
        col_loss = np.sum(np.sum(np.power(offdiag,2),axis=1),axis = 0)
        return col_loss

def pham_loss(AAs):
    n = AAs.shape[1]
    res = 0
    for AA in AAs:
        log_diagdet = np.log(np.linalg.det(np.diag(np.diag(AA))))
        log_det = np.log(np.linalg.det(AA))
        res += log_diagdet - log_det
    return res / (2*n)


def experiment_helper(input_arrays, repeats, with_error, error_levels, \
                      trials, d,n, norm_type, pd= True):
    eps = np.finfo(float).eps
    n_l = len(error_levels)
    times_qndiag, times_rand, times_rand_de, times_pham, times_jade, times_ffdiag,times_uwedge \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)
    errors_qndiag, errors_rand, errors_rand_de, errors_pham, errors_jade, errors_ffdiag, errors_uwedge \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)
    times_rffdiag, errors_rffdiag = np.zeros(n_l), np.zeros(n_l)
    for i, error_level in enumerate(error_levels):
        test_array = random_error(input_arrays,error_level,norm_type,pd)
        for _ in range(repeats):
            start = time()
            Q_rjd = rnojd(test_array,trials,pd=pd)
            #print(np.linalg.norm(np.imag(Q_rjd)))
            end = time()
            times_rand[i] += end - start
            errors_rand[i] += np.sqrt(offdiagonal_frobenius_square(Q_rjd.T @ test_array @  Q_rjd))

        #for _ in range(repeats):
        #    start = time()
        #    B,_,_,_ = uwedge(test_array,)
        #    B = (B.T / np.linalg.norm(B.T,axis=0)).T
        #    end = time()
        #    times_uwedge[i] += end - start
        #    errors_uwedge[i] += np.sqrt(offdiagonal_frobenius_square(B @ test_array @ B.T))
        for _ in range(repeats):
            start = time()
            n = input_arrays.shape[2]
            B0 = np.eye(n)
            B, _ = qndiag(test_array, B0 = B0, ortho=False,  check_sympos = True)  # use the algorithm
            B = (B.T / np.linalg.norm(B.T,axis=0)).T
            end = time()
            times_qndiag[i] += end - start
            errors_qndiag[i] += np.sqrt(offdiagonal_frobenius_square(B @ test_array @ B.T))        

        for _ in range(repeats):
            start = time()
            V,_ = ajd_pham(test_array)
            V = (V.T / np.linalg.norm(V.T,axis=0)).T
            end = time()
            times_pham[i] += end - start
            errors_pham[i] += np.sqrt(offdiagonal_frobenius_square(V @ test_array @ V.T))

        for _ in range(repeats):
            start = time()
            Q_ffdiag, iter = ffdiag(test_array)
            #print("FFDIAG Iter: ",iter)
            Q_ffdiag = (Q_ffdiag.T / np.linalg.norm(Q_ffdiag.T,axis=0)).T
            end = time()
            times_ffdiag[i] += end - start
            errors_ffdiag[i] += np.sqrt(offdiagonal_frobenius_square(Q_ffdiag @ test_array @ Q_ffdiag.T))
        
        for _ in range(repeats):
            start = time()
            Q_rffdiag,iter = ffdiag_rnojd(test_array,trials=1)
            #print("RFFDIAG Iter: ",iter)
            end = time()
            times_rffdiag[i] += end - start
            errors_rffdiag[i] += np.sqrt(offdiagonal_frobenius_square(Q_rffdiag.T @ test_array @ Q_rffdiag))

    title_str = "\\begin{table}[!hbt!]\n" + "\\begin{center}\n" + \
                "\\caption{Runtime and accuracy comparison for " + \
                "$d={d}, n={n}$".format(d = d, n=n) +"}\n" +"\\begin{tabular}{||c|S[table-format=2.2]|c|S[table-format=2.2]|c|S[table-format=2.2]|c||}\n" \
                + "\\hline\n"
    title_str += "Name & Time $\\epsilon_1$ & Error $\epsilon_1$ & Time $\\epsilon_2$ & Error $\\epsilon_2$ &Time $\\epsilon_3$ &Error $\\epsilon_3$\\\\\n"\
         + "\\hline"
    print(title_str)
    print_time_error("FFDIAG", 1000*times_ffdiag / repeats, errors_ffdiag / repeats)
    print_time_error("PHAM", 1000*times_pham / repeats, errors_pham / repeats)
    print_time_error("QNDIAG", 1000*times_qndiag / repeats, errors_qndiag / repeats)
    print_time_error("RSDC", 1000*times_rand / repeats, errors_rand / repeats)
    #print_time_error("RRSDC", 1000*times_rand_de / repeats, errors_rand_de / repeats)
    print_time_error("RFFDIAG", 1000*times_rffdiag / repeats, errors_rffdiag / repeats)
    #print_time_error("UWEDGE", 1000*times_uwedge / repeats, errors_uwedge / repeats)
    print("\\hline")
    if not with_error:
        error_level = 0
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    else:
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    print(closing_str + '\n')

def experiment_helper_ill_conditioned(input_arrays, repeats, with_error, error_levels, \
                      trials, d,n, norm_type):
    test_array = input_arrays
    eps = np.finfo(float).eps
    times_qndiag, times_rand, times_uwedge, times_pham, times_ffdiag \
        = 0, 0, 0, 0, 0,
    errors_qndiag, errors_rand, errors_uwedge, errors_pham, errors_ffdiag \
        = 0, 0, 0, 0, 0,
    times_rffdiag, errors_rffdiag = 0, 0

    for _ in range(repeats):
        start = time()
        Q_rjd = rnojd(test_array,trials,pd = True)
        end = time()
        times_rand += end - start
        errors_rand += np.sqrt(offdiagonal_frobenius_square(Q_rjd.T @ test_array @  Q_rjd))

    for _ in range(repeats):
        start = time()
        n = input_arrays.shape[2]
        B0 = np.eye(n)
        B, _ = qndiag(test_array, B0 = B0, ortho=False, check_sympos = True)  # use the algorithm
        B = (B.T / np.linalg.norm(B.T,axis=0)).T
        end = time()
        times_qndiag += end - start
        errors_qndiag += np.sqrt(offdiagonal_frobenius_square(B @ test_array @ B.T))

    for _ in range(repeats):
        start = time()
        V,_ = ajd_pham(test_array,)
        V = (V.T / np.linalg.norm(V.T,axis=0)).T
        end = time()
        times_pham += end - start
        errors_pham += np.sqrt(offdiagonal_frobenius_square(V @ test_array @ V.T))

    for _ in range(repeats):
        start = time()
        Q_ffdiag, iter = ffdiag(test_array,)
        Q_ffdiag = (Q_ffdiag.T / np.linalg.norm(Q_ffdiag.T,axis=0)).T
        end = time()
        times_ffdiag += end - start
        errors_ffdiag += np.sqrt(offdiagonal_frobenius_square(Q_ffdiag @ test_array @ Q_ffdiag.T))
    for _ in range(repeats):
            start = time()
            Q_rffdiag, iter = ffdiag_rnojd(test_array)
            end = time()
            times_rffdiag += end - start
            errors_rffdiag += np.sqrt(offdiagonal_frobenius_square(Q_rffdiag.T @ test_array @ Q_rffdiag))

    #for _ in range(repeats):
    #        start = time()
    #        B,_,_,_ = uwedge(test_array)
    #        B = (B.T / np.linalg.norm(B.T,axis=0)).T
    #        end = time()
    #        times_uwedge += end - start
    #        errors_uwedge += np.sqrt(offdiagonal_frobenius_square(B @ test_array @ B.T))
    title_str = "\\begin{table}[!hbt!]\n" + "\\begin{center}\n" + \
                "\\caption{Runtime and accuracy comparison for ill-conditioned matrices}\n" +"\\begin{tabular}{||c|c|c||}\n" \
                + "\\hline\n"
    title_str += "Name & Time & Error \\\\\n"\
         + "\\hline"
    print(title_str)
    print_time_error_ill_conditioned("FFDIAG", 1000*times_ffdiag / repeats, errors_ffdiag / repeats)
    print_time_error_ill_conditioned("PHAM", 1000*times_pham / repeats, errors_pham / repeats)
    print_time_error_ill_conditioned("QNDIAG", 1000*times_qndiag / repeats, errors_qndiag / repeats)
    print_time_error_ill_conditioned("RSDC", 1000*times_rand / repeats, errors_rand / repeats)
    print_time_error_ill_conditioned("RFFDIAG", 1000*times_rffdiag / repeats, errors_rffdiag / repeats)
    #print_time_error_ill_conditioned("UWEDGE", 1000*times_uwedge / repeats, errors_uwedge / repeats)
    #print_time_error_ill_conditioned("ManNOJD", 1000*times_manopt / repeats, errors_manopt / repeats)
    print("\\hline")
    if not with_error:
        error_level = 0
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    else:
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    print(closing_str + '\n')

def MA_index(M):
    n = M.shape[0]
    s = 0
    for p in range(n):
        p_col = np.abs(M[:,p])
        p_row = np.abs(M[p,:])
        s+= np.sum(p_col)/ np.max(p_col)
        s+= np.sum(p_row)/ np.max(p_row)
        s-=2
    return s/(2*n*(n-1))
