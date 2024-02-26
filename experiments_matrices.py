from experiments_util import *

def overall_experiments(repeats = 1, trials = 3, MA= False, norm_type = 2, pd = True):
    settings = [(10,100),]
    #settings = [(10,10),(100,10),(10,100),]
    error_levels = [0, 1e-6, 1e-3]
    
    for n,p in settings:
        if pd:
            C = random_jd_matrices(n,p)
        else:
            C = random_jd_matrices_negative(n,p)
        if not MA:
            C = np.array([A / np.linalg.norm(A) for A in C])
            experiment_helper(C, repeats, trials= trials, error_levels=error_levels,\
                            with_error = False, d = n, n = p, norm_type = norm_type, pd = pd)
        np.save('Test_C_'+str(p)+'.npy',C)
overall_experiments(100,3, False, 'fro', True)