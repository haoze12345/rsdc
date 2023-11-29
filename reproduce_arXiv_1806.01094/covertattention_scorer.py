from collections import defaultdict
import h5py
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import os
import pickle
import sys
from utils import rigidsubenv, precompute_covs, score_single_identifier


EXPIDENTIFIER = 'clyde'

PARTITIONSIZE = int(15 * 200)
N_COMPONENTS = None


def run_scorer(parameters):

    expname = parameters.get('expname')
    basepath = './covertattention/' + expname

    # collect already computed scores
    if parameters['identifiers'] == 'collect':
        files = os.listdir(basepath)
        files = [basefile for basefile in files
                 if len(basefile.split('.')) == 5
                 and basefile.split('.')[2] == str(parameters['expsuffix'])
                 and 'score' in basefile]
        scores = [
            pickle.load(open('{}/{}'.format(basepath, fname), 'rb'))
            for fname in files]

    else:
        # Load raw data
        print('Loading raw data...')
        with h5py.File('./covertattention/data.hdf', 'r') as dat:
            X = dat['samples'][...]
            env_index = dat['subject_index'][0, ...]

        dim = X.shape[0]
        if N_COMPONENTS is None:
            n_components = dim

        subenv_index = rigidsubenv(env_index, PARTITIONSIZE)

        # Precompute covmats needed for scoring
        groupcovmats, subenvdiffmats, groups = precompute_covs(
            X,
            env_index,
            subenv_index)

        # Load ICA fits
        print('Loading ICA fits...')

        # determine files with
        # - expsuffix
        files = []
        if len(parameters['identifiers']) > 1:
            files = os.listdir(basepath)
            files = [basefile for basefile in files
                     if len(basefile.split('.')) == 4
                     and basefile.split('.')[2] == str(parameters['expsuffix'])]
        else:
            identifier = parameters['identifiers'][0]
            files = ['{}.{}.{}.pkl'.format(
                len(identifier),
                '_'.join([str(k) for k in np.sort(identifier)]),
                parameters['expsuffix'])]

        data = []
        for onefile in files:
            data.append(pickle.load(open('{}/{}'.format(basepath,
                                                        onefile), 'rb')))

        print('Computing scores...')
        scores = Parallel(
            n_jobs=parameters['n_jobs'], verbose=10)(
                delayed(score_single_identifier)(
                    '{}/{}score.pkl'.format(basepath, fname[:-3]),
                    data_id,
                    env_index,
                    subenv_index,
                    n_components,
                    dim,
                    subenvdiffmats,
                    groupcovmats)
                for fname, data_id in zip(files, data))

    ICAindices = list(scores[0][1].keys())
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for score in scores:
        for ICAindex in ICAindices:
            for inout in ['insample', 'outsample']:
                res[inout][ICAindex][score[0]].extend(
                    score[1][ICAindex][inout])

    # save result
    filename = 'scores_' + str(parameters['expsuffix'])
    print(filename)
    res = {k: dict(v)
           for k, v in res.items()}
    with open(basepath + '/' + filename + '.pkl', 'wb') as f:
        pickle.dump(res, f)


# Main function - runs classification
if __name__ == '__main__':
    # create identifiers
    subs = [k for k in range(8)]
    identifiers = [c
                   for leave_out in range(1, 8)
                   for c in combinations(subs, leave_out)]
    parameters = {}
    parameters['expname'] = EXPIDENTIFIER
    parameters['n_jobs'] = -1
    try:
        if sys.argv[1] == 'collect':
            parameters['identifiers'] = 'collect'
        else:
            parameters['identifiers'] = [identifiers[int(sys.argv[1])]]
    except (IndexError, ValueError):
        parameters['identifiers'] = identifiers
    parameters['expsuffix'] = 'coroICA_base_{}'.format(EXPIDENTIFIER)
    run_scorer(parameters)
