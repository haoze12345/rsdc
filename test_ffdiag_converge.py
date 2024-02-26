import rnojd
from other_jd_algorithms import ffdiag
from experiments_util import *

d = 10
n = 100
C = random_jd_matrices(d,n)
_, iter_rffdiag = ffdiag_rnojd(C)
_, iter_ffdiag = ffdiag(C)
print("# of Iters RFFDIAG: ", iter_rffdiag)
print("# of Iters FFDIAG: ", iter_ffdiag)