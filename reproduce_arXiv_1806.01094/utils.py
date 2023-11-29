from collections import defaultdict
from itertools import chain
import numpy as np
import pickle
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn.base import (BaseEstimator, ClassifierMixin,
                          TransformerMixin, clone, MetaEstimatorMixin)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import lstsq
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.decomposition import PCA


###
# Functions for MD criterion
###
def MDindex(A, Vhat):
    d = np.shape(A)[0]
    G = Vhat.dot(A)
    Gtilde = (G**2) / ((G**2).sum(axis=1)).reshape((d, 1))
    costmat = 1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=0), d).reshape((d, d))
    row_ind, col_ind = linear_sum_assignment(costmat)
    md = np.sqrt(d - np.sum(np.diag(Gtilde[row_ind, col_ind]))) / np.sqrt(d - 1)
    return md


def MDsymm(V1, V2):
    return np.mean([
        MDindex(np.linalg.inv(V1), V2),
        MDindex(np.linalg.inv(V2), V1)])

###
# Helper functions
###

def rigidsubenv(env, nosamples):
    subenv = np.zeros(env.shape)
    for e in np.unique(env):
        subenv[np.where(env == e)] = rigidenv((env == e).sum(),
                                              nosamples) + subenv.max() + 1
    return subenv


def rigidenv(length, nosamples):
    envs = int(np.floor(length / nosamples))
    changepoints = [int(np.round(a))
                    for a in np.linspace(0, length, envs + 1)]
    changepoints = list(set(changepoints))
    changepoints.sort()
    index = np.zeros(length)
    for (i, a, b) in zip(range(envs), changepoints[:-1], changepoints[1:]):
        index[a:b] = i
    return index.reshape(1, -1)


###
# MCIS for sources
###

def precompute_covs(X, env, subenv, pairing='neighbouring'):
    # precompute all covariance matrices of X
    groupcovmats = {}
    subenvcovmats = {}
    subenvcompcovmats = {}
    groups = np.unique(env)

    for group in groups:
        X_group = X[:, env == group]
        subenv_group = subenv[env == group]
        sublist = []
        subcomplist = []
        # precompute all subgroupcovmats
        for subgroup in np.unique(subenv_group):
            sublist.append(np.cov(X_group[:, subenv_group == subgroup]))
            subcomplist.append(np.cov(X_group[:, subenv_group != subgroup]))
        # collect subgroupcovmats
        subenvcovmats[group] = np.copy(sublist)
        # collect subgroupcompcovmats
        if pairing == 'complement':
            subenvcompcovmats[group] = np.copy(subcomplist)
        # precompute groupcovmat
        groupcovmats[group] = np.cov(X_group)

    subenvdiffmats = {}
    for group in groups:
        newlist = []
        if pairing == 'complement':
            for e, f in zip(subenvcovmats[group],
                            subenvcompcovmats[group]):
                newlist.append((e - f))
        elif pairing == 'neighbouring':
            # Note: neighbouring pairing assumes that neighbouring
            # subenvs have neighbouring subenv indices, i.e.,
            # 002211 would break the correct pairing since 01 are
            # not neighbouring subenv indices
            for k in range(len(subenvcovmats[group]) - 1):
                newlist.append((subenvcovmats[group][k] -
                                subenvcovmats[group][k + 1]))
        subenvdiffmats[group] = np.copy(newlist)

    return groupcovmats, subenvdiffmats, groups


def MCIS_scores(V, groupcovmats, subenvdiffmats, env, subinds):
    groups = sorted(list(groupcovmats.keys()))
    offdiags = ~np.eye(V.shape[0], dtype=bool)
    scores = np.repeat(np.nan, len(groups))
    for gind, group in enumerate(groups):
        if group in subinds:
            # Subject-wise score
            scalemat = np.sqrt(np.abs(np.diag(V.dot(groupcovmats[group].dot(V.T)))))
            scalemat = np.outer(scalemat, scalemat)
            scoremat = np.mean([((V.dot(m.dot(V.T))) / scalemat)**2
                                for m in subenvdiffmats[group]], axis=0)
            scores[gind] = np.sqrt(np.mean(scoremat[offdiags]))
    return scores


def score_single_identifier(fname,
                            data,
                            env_index,
                            subenv_index,
                            n_components,
                            dim,
                            subenvdiffmats,
                            groupcovmats):
    trainsubs = data['identifier']
    ICAindices = list(data['V'].keys())

    res = defaultdict(lambda: defaultdict(list))

    for ICAindex in ICAindices:
        print(ICAindex)
        # load ICA trained on training data
        V = data['V'][ICAindex]

        # rownormalise V (does not affect MCIS)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

        traininds = np.isin(env_index, trainsubs)
        trainsub_ind = np.unique(env_index[traininds])
        testsub_ind = np.unique(env_index[~traininds])
        for inout, subinds in zip(
                ['insample', 'outsample'],
                [trainsub_ind, testsub_ind]):
            scores_sub = MCIS_scores(V, groupcovmats, subenvdiffmats,
                                     env_index, subinds)
            res[ICAindex][inout].append(scores_sub)
    with open(fname, 'wb') as f:
        pickle.dump((len(trainsubs), dict(res)), f)
    return (len(trainsubs), dict(res))


###
# Classification
###

class RFsLDA(BaseEstimator, ClassifierMixin):

    def __init__(self, n_sLDA_estimators=200, n_estimators_per_RF=20):
        self.n_sLDA_estimators_ = n_sLDA_estimators
        self.n_estimators_per_RF_ = n_estimators_per_RF
        self.estimator_ = BaggingClassifier(
            base_estimator=IntWeightedShrinkageLDA(),
            n_estimators=n_sLDA_estimators,
            warm_start=True)

    def fit(self, X, y):
        Xc = self.normalise_(X)

        self.estimator_.fit(Xc, y)

        self.estimator_.set_params(
            base_estimator=RandomForestClassifier(
                n_estimators=self.n_estimators_per_RF_,
            ),
            bootstrap=False,
        )

        for k in range(self.n_sLDA_estimators_):
            self.estimator_.set_params(
                n_estimators=self.n_sLDA_estimators_ + 1 + k)
            wrong = self.estimator_.estimators_[k] != y
            self.estimator_.fit(Xc[wrong, :],
                                y[wrong])

    def predict(self, X):
        return self.estimator_.predict(self.normalise_(X))

    def normalise_(self, X):
        return (X - np.median(X, axis=1, keepdims=True)) / \
            np.median(np.abs(X - np.median(X, axis=1, keepdims=True)),
                      axis=1,
                      keepdims=True)


class IntWeightedShrinkageLDA(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.estimator_ = LinearDiscriminantAnalysis(
            shrinkage='auto',
            solver='lsqr')

    def fit(self, X, y, sample_weight):
        sample_weight = np.ceil(sample_weight).astype(int)
        if sample_weight.sum() < 1:
            sample_weight = np.ones(sample_weight.shape)
        inds = list(chain.from_iterable([
            [k] * w
            for k, w in enumerate(sample_weight)]))
        self.estimator_.fit(X[inds, :], y[inds])

    def predict(self, X):
        return self.estimator_.predict(X)


###
# VAR function
###

def lstsq_ram(X, y):
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)
    b = lstsq(XtX, Xty)[0]
    resid = y - X.dot(b)
    return b, resid


def VAR(X, p):
    n, d = np.shape(X)
    Bs = [np.empty((d, d)) for i in range(p)]
    # Compute predictor matrix
    Xpred = np.empty((n-p, d*p))
    for i in range(p):
        Xpred[:, (i*d):((i+1)*d)] = X[(p-i-1):(n-i-1), :]
    # Fit linear regressions for each coordinate
    Btmp, residuals = lstsq_ram(Xpred, X[p:, :])
    Btmp = Btmp.T
    for j in range(p):
        Bs[j] = Btmp[:, (j*d):((j+1)*d)]
    return Bs, residuals


class svarICA(BaseEstimator, TransformerMixin, MetaEstimatorMixin):

    def __init__(self,
                 ICA=None,
                 p=1):
        self.ICA = ICA
        self.p = p

    def fit(self, X, **kwargs):
        X = check_array(X, ensure_2d=True)

        if self.ICA is None:
            ICA = PCA()
        else:
            ICA = clone(self.ICA)

        Bs, residuals = VAR(X, self.p)
        for k in kwargs.keys():
            if type(kwargs[k]) is np.ndarray and kwargs[k].shape == (X.shape[0],):
                kwargs[k] = kwargs[k][self.p:]

        self.ICA_ = ICA.fit(residuals, **kwargs)
        try:
            self.V_ = self.ICA_.V_
        except AttributeError:
            self.V_ = self.ICA_.components_

        self.Bs_ = [self.V_.dot(B)
                    for B in Bs]
        return self

    def transform(self, X):
        check_is_fitted(self, ['Bs_', 'V_', 'ICA_'])
        X = check_array(X)
        residuals = self.V_.dot(X.T)
        residuals[:, :self.p] = 0
        for lag, B in enumerate(self.Bs_):
            residuals[:, self.p:] -= B.dot(X[(self.p-lag-1):-(lag+1), :].T)
        residuals = residuals.T
        return self.ICA_.transform(residuals)
