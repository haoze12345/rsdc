import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter


def butter_highpass(lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=3):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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


def outlierfilling(X, cutoff):
    mads = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    mds = np.median(X, axis=1)
    for (dim, mad, md) in zip(range(X.shape[0]), mads, mds):
        inds = np.where(np.abs(X[dim, :] - md) > cutoff * mad)[0]
        X[dim, inds] = md
    return X


if __name__ == '__main__':
    subs = ['VPiac', 'VPiae', 'VPnh', 'VPmk', 'VPgao', 'VPiah', 'VPiaa', 'VPiai']
    subdata = []
    subtasks = []
    subtrials = []
    for sub in subs:
        mat = loadmat('./covertattention/covertShiftsOfAttention_{}.mat'.format(sub))
        newsub = mat['data']['X'][0][0].T
        # delete EOG channels
        newsub = np.delete(newsub, 6, 0)
        newsub = np.delete(newsub, 1, 0)
        # car, outlier removal, highpass filter
        newsub = car(newsub)
        newsub = outlierfilling(newsub, 10)
        newsub = butter_highpass_filter(newsub,
                                        lowcut=0.5,
                                        fs=200,
                                        order=3)
        subdata.append(newsub)
        # task index
        task = -np.ones((1, newsub.shape[1]))
        trial = -np.ones((1, newsub.shape[1]))
        # trials as described in data description pdf 7.1.
        latent = (mat['mrk']['event'][0][0]['target_latency'][0][0] >= 2000).T
        trialstart = mat['data']['trial'][0][0][latent]
        triallabel = mat['data']['y'][0][0][latent]
        # (8-12 Hz) for the 500-2000 ms interval
        # fs=200
        # interval is from sample 100:400
        count = 0
        for (start, label) in zip(trialstart, triallabel):
            task[0, (start + 100):(start + 400)] = label
            trial[0, (start + 100):(start + 400)] = count
            count += 1
        subtasks.append(task)
        subtrials.append(trial)

    subject_index = np.concatenate([k * np.ones((1, X.shape[1]))
                                    for (k, X) in enumerate(subdata)], axis=1)
    samples = np.concatenate(subdata, axis=1)

    samples = carcomplement(car(samples))

    task_index = np.concatenate(subtasks, axis=1)
    trial_index = np.concatenate(subtrials, axis=1)

    with h5py.File('./covertattention/data.hdf', 'w') as hdf:
        hdf.create_dataset('samples', data=samples, dtype='f')
        hdf.create_dataset('subject_index', data=subject_index, dtype='i')
        hdf.create_dataset('session_index', data=None, dtype='i')
        hdf.create_dataset('task_index', data=task_index, dtype='i')
        hdf.create_dataset('trial_index', data=trial_index, dtype='i')
