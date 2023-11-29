import productionplot
from coroica import CoroICA, UwedgeICA
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import signal
import h5py
import pickle
from utils import precompute_covs, rigidsubenv
from mne.viz import plot_topomap as mne_plot_topomap


def plot_topomap(data, axes, vmin=None, vmax=None):
    fname = './covertattention/montageinfo.pkl'

    with open(fname, 'rb') as f:
        montageinfo = pickle.load(f)

    mne_plot_topomap(data, montageinfo, axes=axes, vmin=vmin, vmax=vmax,
                     show=False, sensors=False)


def carcomplementinv(samples):
    d = samples.shape[0] + 1
    return null(np.ones((1, d))).T.dot(samples)


# returns basis of A's null space
def null(A, eps=1e-15):
    # svd
    u, s, v = np.linalg.svd(A)
    # dimension of null space
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])
    # select columns/rows corresponding to v
    null_mask = np.concatenate(((s <= eps), np.ones((padding, ), dtype=bool)), axis=0)
    null_space = np.compress(null_mask, v, axis=0)
    return null_space


if __name__ == '__main__':
    # Load raw data
    print('Loading raw data...')
    with h5py.File('./covertattention/data.hdf', 'r') as dat:
        X = dat['samples'][...]
        env_index = dat['subject_index'][0, ...]
        trial_index = dat['trial_index'][0, ...]

    # get unmixing matrices
    V = {}

    gica = CoroICA(partitionsize=int(15 * 200),
                   max_iter=10000,
                   tol=0,
                   pairing='neighbouring',
                   rank_components=True)
    gica.fit(X.T, group_index=env_index)
    V['coroICA'] = gica.V_

    fastica = FastICA(random_state=0)
    fastica.fit(X.T)
    V['fastICA'] = fastica.components_

    sobi = UwedgeICA(partitionsize=int(10**6),
                     tol=0,
                     max_iter=10000,
                     condition_threshold=1000,
                     instantcov=False,
                     timelags=list(range(1, 101)))
    sobi.fit(X.T)
    V['SOBI'] = sobi.V_

    selected = {'coroICA': [1, 6, 48]}

    whichsub = 7

    subenv_index = rigidsubenv(env_index, 15 * 200)
    groupcovmats, subenvdiffmats, groups = precompute_covs(
        X,
        env_index,
        subenv_index)
    subenvdiffmats = np.concatenate([v for k, v in subenvdiffmats.items()],
                                    axis=0)

    tmp = np.where(env_index == whichsub)[0]
    a_sub = tmp[0]
    b_sub = tmp[-1]
    a_chunk = a_sub + 25.25 * 200
    b_chunk = a_chunk + 1.75 * 200

    def avgdiffcovV(v, a, subenvdiffmats):
        v = v.reshape(-1, 1)
        curv = np.zeros(v.shape)
        # running average
        for k, m in enumerate(subenvdiffmats):
            newv = m.dot(v)
            if k == 0:
                curv = newv
            else:
                curv += np.sign(v.T.dot(newv)) * newv
        curv = np.sign(np.dot(a, curv)) * curv
        return curv.reshape(-1) / len(subenvdiffmats)

    # create plot of selected components
    plt.close('all')
    fig, axes = plt.subplots(len(selected['coroICA']), 8,
                             gridspec_kw={'width_ratios':
                                          [2.5, 2.5, 2.5, 0.1, 2.5, 0.01, 3.5, 3.5]},
                             dpi=600)

    fig.set_size_inches(productionplot.TEXTWIDTH,
                        productionplot.TEXTWIDTH / 2)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=1.3)

    # compute mixing matrices
    A = {label: carcomplementinv(scipy.linalg.inv(V[label]))
         for label in ['coroICA', 'fastICA', 'SOBI']}

    # find indices best matching the selected coroICA components
    for label in ['fastICA', 'SOBI']:
        corrs = np.abs(np.corrcoef(A['coroICA'].T, A[label].T))
        tmpd = A['coroICA'].shape[1]
        corrs = corrs[:tmpd, tmpd:]
        selected[label] = list(corrs.argmax(axis=1)[selected['coroICA']])

    # build plot
    for compno, ax_row in enumerate(axes):
        for col, ax in enumerate(ax_row):
            if col >= 0 and col <= 2:
                label = ['SOBI', 'fastICA', 'coroICA'][col]
                plot_topomap(np.sign(np.corrcoef(
                    A[label][:, selected[label][compno]],
                    A['coroICA'][:, selected['coroICA'][compno]]
                )[0, 1]) * A[label][:, selected[label][compno]], ax)
                if col == 0:
                    ax.text(-0.8,
                            0.5,
                            [r'\emph{2}$^\mathrm{nd}$',
                             r'\emph{7}$^\mathrm{th}$',
                             r'\emph{49}$^\mathrm{th}$'][compno],
                            verticalalignment='center')
            elif col == 7:
                ax.plot(
                    [k / 200 for k in range(int(b_chunk - a_chunk))],
                    V['coroICA'].dot(
                        X[:, int(a_chunk):int(b_chunk)])[selected['coroICA'][compno], :],
                    linewidth=1.5)
            elif col == 6:
                f, Pxx_den = signal.welch(
                    V['coroICA'].dot(
                        X[:, a_sub:b_sub])[selected['coroICA'][compno], :],
                    200,
                    scaling='spectrum')
                ax.semilogy(f, Pxx_den, linewidth=1.5)
            elif col == 4:
                plot_topomap(carcomplementinv(
                    avgdiffcovV(
                        V['coroICA'].T[:, selected['coroICA'][compno]],
                        scipy.linalg.inv(V['coroICA'])[:, compno],
                        subenvdiffmats)), ax)
            elif col == 3:
                ax.axvline(0, -.1, 1.4, color='black', linewidth=2, clip_on=False)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((1, 1, 1))
            else:
                ax.remove()
            cols = [r'$A_\mathrm{SOBI}$',
                    r'$A_\mathrm{fastICA}$',
                    r'$A_\mathrm{coroICA}$',
                    '',
                    r'$\mathrm{DiffX}(V^\top_\mathrm{coroICA})$',
                    '',
                    r'log power spectrum',
                    r'timeseries chunk']
            if compno == 0 and col <= 4:
                ax.set_title(cols[col], y=1.2)
                if col == 1:
                    ax.text(0, 1.45,
                            'topographies per BSS method',
                            horizontalalignment='center')
            elif compno == 0 and col >= 6:
                ax.set_title(cols[col])
                if col == 6:
                    ax.text(50, 2,
                            'coroICA demixing applied to one subject',
                            horizontalalignment='center')
            elif compno == 2 and col == 6:
                ax.set_xlabel('Frequency in Hz')
            elif compno == 2 and col == 7:
                ax.set_xlabel('Time in sec')

    plt.savefig('./covertattention/Topoplots.pdf', bbox_inches='tight')

    for method in ['SOBI', 'fastICA', 'coroICA']:
        plt.close('all')
        fig, axes = plt.subplots(15, 8,
                                 dpi=600)
        compno = 0
        for axrow in axes:
            for ax1, ax2 in zip(axrow[::2], axrow[1::2]):
                if compno >= 59:
                    ax1.remove()
                    ax2.remove()
                else:
                    if compno in [0, 1, 2, 3]:
                        ax1.set_title(
                            r'$\mathrm{}(V^\top_\mathrm{})$'.format(
                                '{DiffX}',
                                '{' + method + '}'),
                            fontdict={'fontsize': 8})
                        ax2.set_title(r'$A_\mathrm{}$'.format('{' + method + '}'),
                                      fontdict={'fontsize': 8})
                    plot_topomap(carcomplementinv(
                        avgdiffcovV(V[method].T[:, compno],
                                    scipy.linalg.inv(V[method])[:, compno],
                                    subenvdiffmats)), ax1)
                    plot_topomap(A[method][:, compno], ax2)
                    compno += 1
        plt.tight_layout(pad=0, h_pad=0.2, w_pad=0)
        fig.set_size_inches(productionplot.TEXTWIDTH,
                            0.9 * productionplot.TEXTHEIGHT)
        fig.savefig('./covertattention/Topoplots_{}.pdf'.format(method),
                    bbox_inches='tight')
