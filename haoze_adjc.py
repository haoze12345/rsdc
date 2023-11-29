"""Spatial filtering function."""
import warnings
import time

import numpy as np
from scipy.linalg import eigh, inv
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import _check_est, normalize, get_nondiag_weight
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.ajd import ajd_pham
from pyriemann import estimation as est
from pyriemann.preprocessing import Whitening
from scipy.linalg.lapack import dsygv,dggev
from qndiag import qndiag
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np
from rnojd import rnojd, manopt_rnojd, cdopt_rnojd

class RandAJDC(BaseEstimator, TransformerMixin):
    """AJDC algorithm.

    The approximate joint diagonalization of Fourier cospectral matrices (AJDC)
    [1]_ is a versatile tool for blind source separation (BSS) tasks based on
    Second-Order Statistics (SOS), estimating spectrally uncorrelated sources.

    It can be applied:

    * on a single subject, to solve the classical BSS problem [1]_,
    * on several subjects, to solve the group BSS (gBSS) problem [2]_,
    * on several experimental conditions (for eg, baseline versus task), to
      exploit the diversity of source energy between conditions in addition
      to generic coloration and time-varying energy [1]_.

    AJDC estimates Fourier cospectral matrices by the Welch's method, and
    applies a trace-normalization. If necessary, it averages cospectra across
    subjects, and concatenates them along experimental conditions.
    Then, a dimension reduction and a whitening are applied on cospectra.
    An approximate joint diagonalization (AJD) [3]_ allows to estimate the
    joint diagonalizer, not constrained to be orthogonal. Finally, forward and
    backward spatial filters are computed.

    Parameters
    ----------
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.5
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned. Since BSS models assume zero-mean
        processes, the first cospectrum (0 Hz) must be excluded.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.
    dim_red : None | dict, default=None
        Parameter for dimension reduction of cospectra, because Pham's AJD is
        sensitive to matrices conditioning.

        If ``None`` :
            no dimension reduction during whitening.
        If ``{'n_components': val}`` :
            dimension reduction defining the number of components;
            ``val`` must be an integer superior to 1.
        If ``{'expl_var': val}`` :
            dimension reduction selecting the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by ``val``.
            ``val`` must be a float in (0,1], typically ``0.99``.
        If ``{'max_cond': val}`` :
            dimension reduction selecting the number of components such that
            the condition number of the mean matrix is lower than ``val``.
            This threshold has a physiological interpretation, because it can
            be viewed as the ratio between the power of the strongest component
            (usually, eye-blink source) and the power of the lowest component
            you don't want to keep (acquisition sensor noise).
            ``val`` must be a float strictly superior to 1, typically 100.
        If ``{'warm_restart': val}`` :
            dimension reduction defining the number of components from an
            initial joint diagonalizer, and then run AJD from this solution.
            ``val`` must be a square ndarray.
    verbose : bool, default=True
        Verbose flag.

    Attributes
    ----------
    n_channels_ : int
        If fit, the number of channels of the signal.
    freqs_ : ndarray, shape (n_freqs,)
        If fit, the frequencies associated to cospectra.
    n_sources_ : int
        If fit, the number of components of the source space.
    diag_filters_ : ndarray, shape ``(n_sources_, n_sources_)``
        If fit, the diagonalization filters, also called joint diagonalizer.
    forward_filters_ : ndarray, shape ``(n_sources_, n_channels_)``
        If fit, the spatial filters used to transform signal into source,
        also called deximing or separating matrix.
    backward_filters_ : ndarray, shape ``(n_channels_, n_sources_)``
        If fit, the spatial filters used to transform source into signal,
        also called mixing matrix.

    Notes
    -----
    .. versionadded:: 0.2.7

    See Also
    --------
    CospCovariances

    References
    ----------
    .. [1] `On the blind source separation of human electroencephalogram by
        approximate joint diagonalization of second order statistics
        <https://hal.archives-ouvertes.fr/hal-00343628>`_
        M. Congedo, C. Gouy-Pailler, C. Jutten. Clinical Neurophysiology,
        Elsevier, 2008, 119 (12), pp.2677-2686.
    .. [2] `Group indepedent component analysis of resting state EEG in large
        normative samples
        <https://hal.archives-ouvertes.fr/hal-00523200>`_
        M. Congedo, R. John, D. de Ridder, L. Prichep. International Journal of
        Psychophysiology, Elsevier, 2010, 78, pp.89-99.
    .. [3] `Joint approximate diagonalization of positive definite
        Hermitian matrices
        <https://epubs.siam.org/doi/10.1137/S089547980035689X>`_
        D.-T. Pham. SIAM Journal on Matrix Analysis and Applications, Volume 22
        Issue 4, 2000
    """

    def __init__(self, window=128, overlap=0.5, fmin=None, fmax=None, fs=None,
                 dim_red=None, verbose=True):
        """Init."""

        self.window = window
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.dim_red = dim_red
        self.verbose = verbose

    def fit(self, X, y=None, algorithm=None, need_transpose = True):
        """Fit.

        Compute and diagonalize cospectra, to estimate forward and backward
        spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_subjects, n_conditions, n_channels, n_times) | \
                list of n_subjects of list of n_conditions ndarray of shape \
                (n_channels, n_times), with same n_conditions and n_channels \
                but different n_times
            Multi-channel time-series in channel space, acquired for different
            subjects and under different experimental conditions.
        y : None
            Currently not used, here for compatibility with sklearn API.

        Returns
        -------
        self : AJDC instance
            The AJDC instance.
        """
        # definition of params for Welch's method
        cospcov = est.CospCovariances(
            window=self.window,
            overlap=self.overlap,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs)
        # estimation of cospectra on subjects and conditions
        cosp = []
        for s in range(len(X)):
            cosp_ = cospcov.transform(X[s])
            if s == 0:
                n_conditions = cosp_.shape[0]
                self.n_channels_ = cosp_.shape[1]
                self.freqs_ = cospcov.freqs_
            else:
                if n_conditions != cosp_.shape[0]:
                    raise ValueError('Unequal number of conditions')
                if self.n_channels_ != cosp_.shape[1]:
                    raise ValueError('Unequal number of channels')
            cosp.append(cosp_)
        cosp = np.transpose(np.array(cosp), axes=(0, 1, 4, 2, 3))

        # trace-normalization of cospectra, Eq(3) in [2]
        cosp = normalize(cosp, "trace")
        # average of cospectra across subjects, Eq(7) in [2]
        cosp = np.mean(cosp, axis=0, keepdims=False)
        # concatenation of cospectra along conditions
        self._cosp_channels = np.concatenate(cosp, axis=0)
        # estimation of non-diagonality weights, Eq(B.1) in [1]
        weights = get_nondiag_weight(self._cosp_channels)

        # initial diagonalizer: if warm restart, dimension reduction defined by
        # the size of the initial diag filters
        init = None
        if self.dim_red is None:
            warnings.warn('Parameter dim_red should not be let to None')
        elif isinstance(self.dim_red, dict) and len(self.dim_red) == 1 \
                and next(iter(self.dim_red)) == 'warm_restart':
            init = self.dim_red['warm_restart']
            if init.ndim != 2 or init.shape[0] != init.shape[1]:
                raise ValueError(
                    'Initial diagonalizer defined in dim_red is not a 2D '
                    'square matrix (Got shape = %s).' % (init.shape,))
            self.dim_red = {'n_components': init.shape[0]}

        # dimension reduction and whitening, Eq.(8) in [2], computed on the
        # weighted mean of cospectra across frequencies (and conditions)
        whit = Whitening(
            metric='euclid',
            dim_red=self.dim_red,
            verbose=self.verbose)
        cosp_rw = whit.fit_transform(self._cosp_channels, weights)
        self.n_sources_ = whit.n_components_

        # approximate joint diagonalization, currently by Pham's algorithm [3]
        time_start = time.time()
        if algorithm == None:
            self.diag_filters_ = manopt_rnojd(cosp_rw).T
        else:
            self.diag_filters_ = algorithm(cosp_rw)
            if need_transpose:
                self.diag_filters_ = self.diag_filters_.T
        time_spent_jd = time.time() - time_start

        self._cosp_sources = self.diag_filters_ @ cosp_rw @ self.diag_filters_.T
        # computation of forward and backward filters, Eq.(9) and (10) in [2]
        self.forward_filters_ = self.diag_filters_ @ whit.filters_.T
        self.backward_filters_ = whit.inv_filters_.T @ inv(self.diag_filters_)
        return self, time_spent_jd

    def transform(self, X):
        """Transform channel space to source space.

        Transform channel space to source space, applying forward spatial
        filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.

        Returns
        -------
        source : ndarray, shape (n_matrices, n_sources, n_times)
            Multi-channel time-series in source space.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                'X does not have the good number of channels. Should be %d but'
                ' got %d.' % (self.n_channels_, X.shape[1]))

        source = self.forward_filters_ @ X
        return source

    def inverse_transform(self, X, supp=None):
        """Transform source space to channel space.

        Transform source space to channel space, applying backward spatial
        filters, with the possibility to suppress some sources, like in BSS
        filtering/denoising.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_sources, n_times)
            Multi-channel time-series in source space.
        supp : list of int | None, default=None
            Indices of sources to suppress. If None, no source suppression.

        Returns
        -------
        signal : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_sources_:
            raise ValueError(
                'X does not have the good number of sources. Should be %d but '
                'got %d.' % (self.n_sources_, X.shape[1]))

        denois = np.eye(self.n_sources_)
        if supp is None:
            pass
        elif isinstance(supp, list):
            for s in supp:
                denois[s, s] = 0
        else:
            raise ValueError('Parameter supp must be a list of int, or None')

        signal = self.backward_filters_ @ denois @ X
        return signal

    def get_src_expl_var(self, X):
        """Estimate explained variances of sources.

        Estimate explained variances of sources, see Appendix D in [1].

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.

        Returns
        -------
        src_var : ndarray, shape (n_matrices, n_sources)
            Explained variance for each source.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                'X does not have the good number of channels. Should be %d but'
                ' got %d.' % (self.n_channels_, X.shape[1]))

        cov = est.Covariances().transform(X)

        src_var = np.zeros((X.shape[0], self.n_sources_))
        for s in range(self.n_sources_):
            src_var[:, s] = np.trace(
                self.backward_filters_[:, s] * self.forward_filters_[s].T * cov
                * self.forward_filters_[s] * self.backward_filters_[:, s].T,
                axis1=-2,
                axis2=-1)
        return src_var