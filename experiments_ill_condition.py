from experiments_util import *

def overall_experiments(repeats = 100, trials = 3, MA= False, norm_type = 2):
    settings = [(20,30)]
    error_levels = [0,0,0]
    for n,p in settings:
        C = random_jd_ill_conditioned_matrices(n,p)
        C = np.array([A / np.linalg.norm(A) for A in C])
        if not MA:
            experiment_helper_ill_conditioned(C, repeats, trials= trials, error_levels=error_levels,\
                            with_error = False, d = n, n = p, norm_type = norm_type)
overall_experiments(100,3, False, 'fro')