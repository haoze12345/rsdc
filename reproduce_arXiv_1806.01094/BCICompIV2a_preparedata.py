import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import trange


def car(samples):
    d = samples.shape[0]
    centering = np.eye(d) - np.ones((d, d)) / d
    return centering.dot(samples)


# returns basis of A's null space
def null(A, eps=1e-15):
    # svd
    u, s, v = np.linalg.svd(A)
    # dimension of null space
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])
    # select columns/rows corresponding to v
    null_mask = np.concatenate(((s <= eps),
                                np.ones((padding,), dtype=bool)), axis=0)
    null_space = np.compress(null_mask, v, axis=0)
    return null_space


def carcomplement(samples):
    d = samples.shape[0]
    carcomp = null(np.ones((1, d)))
    return carcomp.dot(samples)


def trimmed_mean(x):
    sorts = np.argsort(np.abs(x - x.mean(axis=0, keepdims=True)), axis=0)[:-1, :]
    return np.mean(x[sorts, np.arange(x.shape[1])], axis=0)


def trimmed_means(X):
    electrode_groups = [
        [6, 1, 7, 13],
        [0, 2, 3, 4],
        [5, 11, 17, 12],
        [21, 18, 19, 20],
        [8, 9, 10, 14, 15, 16]
    ]
    new = np.empty((len(electrode_groups), X.shape[1]))
    for k, group in enumerate(electrode_groups):
        new[k, :] = trimmed_mean(X[group, :])
    return new


if __name__ == '__main__':
    subs = ['A01T', 'A01E', 'A02T', 'A02E', 'A03T', 'A03E', 'A04T', 'A04E',
            'A05T', 'A05E', 'A06T', 'A06E', 'A07T', 'A07E', 'A08T', 'A08E',
            'A09T', 'A09E']

    # sampled at 250 Hz, bandpass-filtered 0.5--100 Hz5, 50 Hz notch-filtered
    # extract seconds 3--6
    # Assumptions:
    #     * last three channels are EOG!
    #     * timestamps are fixation cross onset
    samples = np.zeros((22, 0))
    subject_index = np.zeros((1, 0))
    session_index = np.zeros((1, 0))
    trial_index = np.zeros((1, 0))
    task_index = np.zeros((1, 0))
    trialno = 0
    sessionno = 0
    for sub in trange(len(subs), desc='extract subject trials'):
        sub = subs[sub]
        mat = loadmat('./BCICompIV2a/{}.mat'.format(sub))
        startses = 3
        # since A04T has two EOG blocks missing
        if sub == 'A04T':
            startses = 1
        for ses in range(startses, mat['data'].shape[1]):
            times = mat['data'][0, ses][0][0][1].reshape(-1) + 3 * 250
            timeinds = [k + l for k in times for l in range(3 * 250)]
            raw = car(mat['data'][0, ses][0][0][0][:, :-3].T)
            triallab = mat['data'][0, ses][0][0][2].repeat(3 * 250).reshape(1, -1)
            lab = -np.ones((1, raw.shape[1]))
            lab[:, timeinds] = triallab
            samples = np.concatenate([samples, raw], axis=1)
            subject_index = np.concatenate(
                [subject_index, subs.index(sub) * np.ones(lab.shape)], axis=1)
            session_index = np.concatenate(
                [session_index, sessionno * np.ones(lab.shape)], axis=1)
            notrials = mat['data'][0, ses][0][0][2].shape[0]
            new_trials = -np.ones((1, raw.shape[1]))
            new_trials[:, timeinds] = trialno + np.arange(notrials).repeat(
                3 * 250).reshape(1, -1)
            trial_index = np.concatenate(
                [trial_index,
                 new_trials], axis=1)
            task_index = np.concatenate([task_index, lab], axis=1)
            trialno = np.max(new_trials) + 1
            sessionno += 1

    samples = carcomplement(samples)

    with h5py.File('./BCICompIV2a/data.hdf', 'w') as hdf:
        hdf.create_dataset('samples', data=samples, dtype='f')
        hdf.create_dataset('subject_index', data=subject_index, dtype='i')
        hdf.create_dataset('session_index', data=session_index, dtype='i')
        hdf.create_dataset('trial_index', data=trial_index, dtype='i')
        hdf.create_dataset('task_index', data=task_index, dtype='i')

    with h5py.File('./BCICompIV2a/data.hdf', 'r+') as hdf:
        # this fixes the wrong subject/session indices (2 recording days each)
        subject_index = hdf['subject_index'][...]
        session_index = hdf['session_index'][...]
        for k in range(1, 18, 2):
            subject_index[subject_index == k] = k - 1
        hdf.__delitem__('subject_index')
        hdf.__delitem__('session_index')
        hdf['subject_index'] = subject_index
        hdf['session_index'] = session_index
