"""Spatial filtering function."""
import numpy as np
import scipy as sp
from scipy.linalg.lapack import dsygv,dggev
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import cdopt
from other_jd_algorithms import ffdiag, qndiag

def runif_in_simplex(AA):
    n = AA.shape[0]
    k = np.random.exponential(scale=1.0, size=n)
    k /= sum(k)
    A_mu = np.einsum('ijk,i->jk',AA, k)
    return A_mu

def random_combination(AA):
    """
    Diagonlize a random linear combination of an array of matrices
    AA: of shape (d,n,n)
    d: # of matrices

    return: the orthognoal matrix Q
    """
    mu = np.random.normal(0,1,AA.shape[0])
    A_mu = np.einsum('ijk,i->jk',AA, mu)
    return A_mu

def offdiag(A):
    shape = A.shape
    identity_3d = np.zeros(shape)
    idx = np.arange(shape[1])
    identity_3d[:, idx, idx] = 1 
    mask = np.array(1 - identity_3d, dtype = np.int0)
    offdiag = A * mask
    return offdiag

def offdiag_frobenius_square(A):
    loss = np.sum(np.square(offdiag(A)))
    return loss

def rnojd(C,trials=5, pd = True):
    min_offdiag = np.inf
    best_X = None
    for _ in range(trials):
        A_mu = random_combination(C)
        if pd:
            A_theta = np.mean(C,axis=0)
        else:
            A_theta = random_combination(C)
        _, _, _, _, X, _, _ = dggev(A_mu,A_theta)
        X_norm = X / np.linalg.norm(X,axis=0)
        current_offdiag = offdiag_frobenius_square(X_norm.T @ C @ X_norm)
        if trials == 1:
            return X_norm
        if current_offdiag < min_offdiag:
            best_X = X_norm
            min_offdiag = current_offdiag
    return best_X

def ffdiag_rnojd(C, max_iter=10, trials=1, pd = True,):
    initial_point=rnojd(C,trials, pd = pd).T
    X, iter = ffdiag(C,V0 = initial_point, max_iter=max_iter)
    X = X.T
    X_norm = X / np.linalg.norm(X,axis=0)
    return X_norm, iter


def manopt_rnojd_old(C, trials=5, max_iter = 100, trivial_init=False, pd = False, beta_rule = "HestenesStiefel"):
    manifold = pymanopt.manifolds.Oblique(C.shape[1],C.shape[1])
    def create_cost_and_derivates():
        @pymanopt.function.autograd(manifold)
        def cost(X):
            return offdiag_frobenius_square(X.T@C@X)

        @pymanopt.function.numpy(manifold)
        def riemannian_gradient(X):
            transformed_C = X.T@C@X
            identity_3d = np.zeros(C.shape)
            idx = np.arange(C.shape[1])
            identity_3d[:, idx, idx] = 1 
            mask = np.array(1 - identity_3d, dtype = np.int0)
            offdiag = transformed_C * mask
            gradient_matrices = C @ X @ offdiag
            euclidean_gradient = 2 * np.sum(gradient_matrices, axis=0)
            return euclidean_gradient - X.T@np.diag(np.diag(X.T@euclidean_gradient))
        
        '''@pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            identity_3d = np.zeros(C.shape)
            idx = np.arange(C.shape[1])
            identity_3d[:, idx, idx] = 1 
            mask = np.array(1 - identity_3d, dtype = np.int0)
            offdiag = (X.T@C@X) * mask
            gradient_matrices = C @ X @ offdiag
            euclidean_gradient = 2 * np.sum(gradient_matrices, axis=0)
            return euclidean_gradient'''

        """@pymanopt.function.autograd(manifold)
        def euclidean_hessian(X,Z):
            identity_3d = np.zeros(C.shape)
            idx = np.arange(C.shape[1])
            identity_3d[:, idx, idx] = 1 
            mask = np.array(1 - identity_3d, dtype = np.int0)
            offdiag_XX = 2*np.sum(C@Z@(X.T@C@X) * mask, axis=0)
            offdiag_ZX = 2*np.sum(C@X@(Z.T@C@X) * mask, axis=0)
            offdiag_XZ = 2*np.sum(C@X@(X.T@C@Z) * mask, axis=0)
            return offdiag_XX + offdiag_ZX + offdiag_XZ"""

        return cost, riemannian_gradient, #euclidean_gradient
    
    cost, riemannian_gradient= create_cost_and_derivates()
    problem = pymanopt.Problem(
            manifold,
            cost,
            riemannian_gradient=riemannian_gradient,
    )

    optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations=max_iter,verbosity=0,
                                                      log_verbosity=0,
                                                      line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher(),
                                                      beta_rule = beta_rule
                                                      )
    if trivial_init:
        result = optimizer.run(problem, initial_point=None)
    else:
        result = optimizer.run(problem, initial_point=rnojd(C,trials, pd = pd))
    #print("Final Error:", result.cost)
    #print("Stopping Criteion:", result.stopping_criterion )
    return result.point

def manopt_rnojd(C, trials=5, max_iter = 100, trivial_init=False, pd = False, beta_rule = "HestenesStiefel"):
    manifold = pymanopt.manifolds.Oblique(C.shape[1],C.shape[1])
    transformed_C = None
    offdiag_transformed = None
    @pymanopt.function.autograd(manifold)
    def cost(X):
        global transformed_C, offdiag_transformed

        transformed_C = X.T @ C @ X
        offdiag_transformed = offdiag(transformed_C)
        return np.sum(np.square(offdiag_transformed))
    
    @pymanopt.function.numpy(manifold)
    def riemannian_gradient(X):
            global transformed_C, offdiag_transformed
            gradient_matrices = C @ X @ offdiag_transformed
            euclidean_gradient = 2 * np.sum(gradient_matrices, axis=0)
            return euclidean_gradient - X.T@np.diag(np.diag(X.T@euclidean_gradient))
    
    problem = pymanopt.Problem(
            manifold,
            cost,
            riemannian_gradient=riemannian_gradient,
    )

    optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations=max_iter,verbosity=0,
                                                      log_verbosity=0,
                                                      line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher(),
                                                      beta_rule = beta_rule
                                                      )
    if trivial_init:
        result = optimizer.run(problem, initial_point=None)
    else:
        result = optimizer.run(problem, initial_point=rnojd(C,trials, pd = pd))
    return result.point

def cdopt_rnojd(C, trials =3, opt_method = 'L-BFGS-B'):

    def obj_fun(X):
        return offdiag_frobenius_square(X.T@C@X)

    def obj_grad(X):
        identity_3d = np.zeros(C.shape)
        idx = np.arange(C.shape[1])
        identity_3d[:, idx, idx] = 1 
        mask = np.array(1 - identity_3d, dtype = np.int0)
        offdiag = (X.T@C@X) * mask
        gradient_matrices = C @ X @ offdiag
        euclidean_gradient = 2 * np.sum(gradient_matrices, axis=0)
        return euclidean_gradient
    
    M = cdopt.manifold_np.oblique_np((C.shape[1],C.shape[1]))   # The Oblique manifold.
    problem_test = cdopt.core.problem(M, obj_fun, obj_grad=obj_grad, beta = 'auto')  # describe the optimization problem and set the penalty parameter \beta.
    # the vectorized function value, gradient and Hessian-vector product of the constraint dissolving function. Their inputs are numpy 1D array, and their outputs are float or numpy 1D array.
    cdf_fun_np = problem_test.cdf_fun_vec_np
    cdf_grad_np = problem_test.cdf_grad_vec_np 
    Xinit = M.m2v(rnojd(C,trials))
    out_msg = sp.optimize.minimize(cdf_fun_np, Xinit ,method=opt_method,jac = cdf_grad_np, options={'disp': None, 'maxcor': 10, 'ftol': 0, 'gtol': 1e-06, 'eps': 0e-08,})
    X = M.v2m(out_msg.x)
    X_norm = X / np.linalg.norm(X,axis=0)
    return X_norm