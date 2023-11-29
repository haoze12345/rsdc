import h5py
import numpy as np
import pickle
from itertools import combinations
import sklearn
from sklearn.decomposition import FastICA, PCA
from coroica import CoroICA, UwedgeICA
import sys
import gc


EXPIDENTIFIER = 'clyde'

eps = 0
n_iter_max = 10000
minimize_loss = False
n_components = None

PARTITIONSIZE = int(15 * 200)
PARTITIONSIZES = [int(k * 200)
                  for k in [15, 30, 60]]

lags = [1, 2, 3, 4]


def ICA_METHODS():
    return (
        ('coroICA_base_{}'.format(EXPIDENTIFIER),
         CoroICA(partitionsize=PARTITIONSIZE,
                 tol=eps,
                 max_iter=n_iter_max,
                 n_components=n_components,
                 minimize_loss=minimize_loss,
                 condition_threshold=1000,
                 pairing='neighbouring')),
        ('uwedgeICA_var',
         UwedgeICA(partitionsize=PARTITIONSIZE,
                   tol=eps,
                   max_iter=n_iter_max,
                   n_components=n_components,
                   minimize_loss=minimize_loss,
                   condition_threshold=1000,
                   instantcov=True,
                   timelags=None)),
        ('uwedgeICA_var-TD',
         UwedgeICA(partitionsize=PARTITIONSIZE,
                   tol=eps,
                   max_iter=n_iter_max,
                   n_components=n_components,
                   minimize_loss=minimize_loss,
                   condition_threshold=1000,
                   instantcov=True,
                   timelags=lags)),
        ('uwedgeICA_TD',
         UwedgeICA(partitionsize=PARTITIONSIZE,
                   tol=eps,
                   max_iter=n_iter_max,
                   n_components=n_components,
                   minimize_loss=minimize_loss,
                   condition_threshold=1000,
                   instantcov=False,
                   timelags=lags)),
        ('uwedgeICA_SOBI',
         UwedgeICA(partitionsize=int(10**6),
                   tol=eps,
                   max_iter=n_iter_max,
                   n_components=n_components,
                   minimize_loss=minimize_loss,
                   condition_threshold=1000,
                   instantcov=False,
                   timelags=list(range(1, 101)))),
        ('fastICA', FastICA(n_components=n_components)),
    )


# Function that computes ICAs for a single identifier (set of subjects)
# and saves unmixing matrics
def runICAs(identifier,
            expname,
            samples,
            subject_index,
            ICAs):

    # determine filename
    fname = '{}.{}.{}.pkl'.format(
        len(identifier),
        '_'.join([str(k) for k in np.sort(identifier)]),
        ICAs[0][0]
    )

    # construct boolean trainset index
    trainset = np.isin(subject_index, identifier)
    Xtrain = samples[:, trainset]
    trainsubject_index = subject_index[trainset]

    # iterate over ICAs
    V = {}
    failedICAs = []
    for idx in range(len(ICAs)):
        print(ICAs[idx][0])
        try:
            ica = sklearn.clone(ICAs[idx][1])
            if 'coroICA' in ICAs[idx][0]:
                ica.fit(Xtrain.T,
                        group_index=trainsubject_index)
            else:
                ica.fit(Xtrain.T)

            try:
                Vtmp = ica.V_
            except AttributeError:
                Vtmp = ica.components_
        except Exception:
            failedICAs.append(ICAs[idx][0])
            Vtmp = np.random.randn(Xtrain.shape[0], Xtrain.shape[0])
            if n_components is not None:
                Vtmp = Vtmp[n_components, :]

        V[ICAs[idx][0]] = np.copy(Vtmp)

        gc.collect()

    res = {'identifier': identifier,
           'failedICAs': failedICAs,
           'V': V,
           'ICAinfo': ICAs}

    with open('./covertattention/' + expname + '/' + fname, 'wb') as f:
        pickle.dump(res, f)


# Function that runs a single experiment
def run_experiment(parameters):
    # Load preprocessed data
    with h5py.File('./covertattention/data.hdf', 'r') as dat:
        X = dat['samples'][...]
        env_index = dat['subject_index'][0, ...]

    # Read out parameters
    if parameters.get('identifier', None) is None:
        raise RuntimeError('no identifier given')
    else:
        identifier = parameters.get('identifier')
    ICAs = parameters.get('ICAs')
    expname = parameters.get('expname')

    # Run all ICAs and save
    runICAs(identifier,
            expname,
            X,
            env_index,
            ICAs)


# Main function - Runs all experiments
if __name__ == '__main__':
    try:
        which_identifier = sys.argv[1]
    except IndexError:
        print('Please provide argument #identifier '
              'to select which of the 254 identifiers to run.')
        raise

    # create identifiers
    subs = [k for k in range(8)]
    identifiers = [c
                   for leave_out in range(1, 8)
                   for c in combinations(subs, leave_out)]
    parameters = {}
    parameters['expname'] = EXPIDENTIFIER
    parameters['identifier'] = identifiers[int(which_identifier)]
    parameters['ICAs'] = ICA_METHODS()
    run_experiment(parameters)
