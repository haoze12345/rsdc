import productionplot
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import binom
sns.set_style('darkgrid')


EXPIDENTIFIER = 'clyde'


def plot(parameters={}):

    basepath = './covertattention/' + parameters['expname']

    filenames = basepath + '/classification_data_' + \
        '.' + str(parameters['expsuffix']) + '_{}_trainsubs.pkl'

    exps = list(range(1, 8))

    xlabel = 'number of training subjects'
    xshift = 0.4
    width = 0.15

    methods = [
        ['coroICA_base_{}'.format(EXPIDENTIFIER), 'coroICA'],
        ['fastICA', 'fastICA'],
        ['uwedgeICA_var', 'choiICA (var)'],
        ['uwedgeICA_var-TD', 'choiICA (TD\&var)'],
        ['uwedgeICA_TD', 'choiICA (TD)'],
        ['uwedgeICA_SOBI', 'SOBI'],
    ]

    colors = productionplot.colormapping(methods)

    methods_keys = [k[0] for k in methods]
    methods_labels = [k[1] for k in methods]

    centerpos = np.arange(1, len(exps) + 1)
    positions = [[c + shift
                  for c in centerpos]
                 for shift in np.linspace(- xshift,
                                          xshift,
                                          len(methods_keys) + 1)]
    # Initialize plot
    fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 2.5))

    ax = plt.subplot()

    # Prepare data
    vios = []
    for metkey in methods_keys:
        nextmet = []
        for exp in exps:
            pkl = pickle.load(open(filenames.format(exp), 'rb'))
            nextmet.append(np.nanmean(pkl[metkey], axis=0).mean(axis=1))
        vios.append(nextmet)

    # Construct plot
    violins = []
    for metkey, vio, pos in zip(methods_keys, vios, positions):
        violins.append(ax.violinplot(
            vio,
            pos,
            points=100,
            widths=width,
            showmeans=False,
            showextrema=True,
            showmedians=True))

    for vio, col in zip(violins, colors[:len(violins)]):
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = vio[partname]
            vp.set_edgecolor(col)
        for vp in vio['bodies']:
            vp.set_facecolor(col)
            vp.set_edgecolor(col)

    plt.ylabel('average classification accuracy [\%]')

    ax.axhline(1 / 6, color='gray', linewidth=1, zorder=1)

    for posA, posB, exp in zip(positions[0], positions[-1], exps):
        no_trials_heldout = (8 - exp) * 311
        CIinterval = np.asarray(
            binom.interval(.95, no_trials_heldout, 1 / 6)) / \
            no_trials_heldout
        print(CIinterval)
        ax.add_patch(mpatches.Rectangle(
            (posA - xshift / (len(methods)), CIinterval[0]),
            posB - posA,
            CIinterval[1] - CIinterval[0],
            color='gray',
            alpha=0.25,
            zorder=1,
            linewidth=0
        ))

    ax.set_xticks(np.arange(1, len(exps) + 1))
    ax.set_xticklabels(['{}'.format(exp) for exp in exps])
    plt.xlabel(xlabel)
    patches = [
        mpatches.Patch(color=vio['bodies'][0].get_facecolor()[0])
        for vio in violins]
    lgd = ax.legend(patches, methods_labels,
                    prop={'size': 6},
                    loc='center left', bbox_to_anchor=(1, 0.5))

    fig.set_size_inches(productionplot.TEXTWIDTH, 2.5)

    plt.savefig(filenames[:-17] + '.pdf',
                bbox_inches='tight',
                bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    parameters = {}
    parameters['expname'] = EXPIDENTIFIER
    parameters['expsuffix'] = 'coroICA_base_{}'.format(EXPIDENTIFIER)

    plot(parameters)
