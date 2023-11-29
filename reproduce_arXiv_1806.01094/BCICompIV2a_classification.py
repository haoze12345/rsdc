from collections import defaultdict
import h5py
import numpy as np
import os
import pickle
from joblib import Parallel, delayed
from scipy.signal import butter, lfilter
from sklearn.utils import resample
from utils import RFsLDA


EXPIDENTIFIER = 'bonnie'


###
# Classification
###

def run_classification(parameters):

    # read out parameters
    expname = parameters.get('expname')

    # Load raw data
    print('Loading raw data...')
    with h5py.File('./BCICompIV2a/data.hdf', 'r') as dat:
        X = dat['samples'][...]
        env_index = dat['subject_index'][0, ...]
        trial_index = dat['trial_index'][0, ...]
        task_index = dat['task_index'][0, ...]
        triallength = int(3 * 250)

    # Load ICA fits
    print('Loading ICA fits...')
    basepath = './BCICompIV2a/' + expname

    for no_trainsubs in range(1, 9):
        # determine files with
        # - no_trainsubs number of subs used for training
        # - expsuffix
        files = os.listdir(basepath)
        files = [basefile for basefile in files
                 if basefile.split('.')[0] == str(no_trainsubs)
                 and basefile.split('.')[2] == str(parameters['expsuffix'])
                 and 'score' not in basefile]
        data = []
        for onefile in files:
            data.append(pickle.load(open('{}/{}'.format(basepath,
                                                        onefile), 'rb')))

        parares = Parallel(
            n_jobs=parameters['n_jobs'], verbose=10)(
                delayed(single_identifier)(
                    data_id,
                    env_index,
                    task_index,
                    trial_index,
                    X,
                    triallength,
                    parameters['filterspecs'])
                for data_id in data)

        ICAindices = list(data[0]['V'].keys())

        result_final = defaultdict(list)
        for res in parares:
            for icd in ICAindices:
                result_final[icd].append(res[icd])

        # print overview
        means_over_identifier = {
            icd: np.nanmean(np.vstack(result_final[icd]), axis=0)
            for icd in result_final.keys()}
        for k, v in means_over_identifier.items():
            print('{: <25}: {:.3f}'.format(k, np.nanmean(v)))

        # save result
        filename = 'classification_data_' + \
            '.' + str(parameters['expsuffix']) + '_{}_trainsubs'.format(
                no_trainsubs)
        print(filename)
        with open(basepath + '/' + filename + '.pkl', 'wb') as f:
            pickle.dump(result_final, f)


# Required helper functions for run_classification
def single_identifier(data,
                      env_index,
                      task_index,
                      trial_index,
                      X,
                      triallength,
                      filterspecs):
    ICAindices = list(data['V'].keys())

    cv_test = defaultdict(list)

    # continuous data for training
    trainset = np.isin(env_index, data['identifier'])
    # trial data for testing
    traintrials = (trainset) & (trial_index > -1)

    print(data['identifier'])

    # which feature function and classifier to use
    # bandpassfilter log variance where filters are set per dataset
    def feat(dat, envid, trialid):
        return features(tensorfilter(dat,
                                     filterspecs[0],
                                     filterspecs[1],
                                     filterspecs[2],
                                     filterspecs[3]), envid, trialid)

    label_train = task_index[traintrials][::triallength]

    for ICAindex in ICAindices:
        print(ICAindex)
        Vtrain = data['V'][ICAindex]

        Shat = Vtrain.dot(X)

        for env in np.unique(env_index):
            Shat[:, env_index == env] = Shat[:, env_index == env] / \
                Shat[:, env_index == env].std(axis=1, keepdims=True)

        # Compute features
        fv_train = feat(Shat[:, traintrials],
                        env_index[traintrials],
                        trial_index[traintrials])

        # Get the score several times to estimate the variance etc
        for repetition in range(50):
            model = RFsLDA()
            model.fit(
                resample(fv_train.T, random_state=repetition),
                resample(label_train, random_state=repetition)
            )
            score_res = [np.nan] * len(np.unique(env_index))
            for eno, e in enumerate(np.unique(env_index)):
                if e not in data['identifier']:
                    inds = (trial_index > -1) & (env_index == e)
                    score_res[eno] = model.score(
                        resample(
                            feat(
                                Shat[:, inds],
                                env_index[inds],
                                trial_index[inds]).T,
                            random_state=repetition),
                        resample(
                            task_index[inds][::triallength],
                            random_state=repetition))
            print('{}: {}'.format(ICAindex, score_res))
            cv_test[ICAindex].append(score_res)
    return cv_test


def tensorfilter(X, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    if lowcut is not None and highcut is not None:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
    elif highcut is None:
        low = lowcut / nyq
        b, a = butter(order, low, btype='high')
    elif lowcut is None:
        high = highcut / nyq
        b, a = butter(order, high, btype='low')
    return lfilter(b, a, X, axis=-1)


def features(samples, env_index, trial_index):
    fv = np.empty((samples.shape[0], 0))
    envs = np.unique(env_index)
    for idx, env in enumerate(envs):
        trial_index_env = trial_index[env_index == env]
        trials = np.unique(trial_index_env)
        samples_env = samples[:, env_index == env]
        for idy, trial in enumerate(trials):
            fv = np.hstack([fv, np.ma.log(np.var(samples_env[:, trial_index_env == trial],
                                                 axis=1,
                                                 keepdims=True)).filled(0)])
    return fv


# Main function - runs classification
if __name__ == '__main__':

    # Classification experiment
    parameters = {}
    parameters['filterspecs'] = [8, 30, 250, 3]
    parameters['n_jobs'] = -1
    parameters['expname'] = EXPIDENTIFIER
    parameters['expsuffix'] = 'coroICA_base_{}'.format(EXPIDENTIFIER)
    run_classification(parameters)
