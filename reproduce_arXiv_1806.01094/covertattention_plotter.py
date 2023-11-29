import productionplot
import numpy as np
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


EXPIDENTIFIER = 'clyde'


def plot(parameters, paired=False):

    basepath = './covertattention/' + parameters['expname']
    filename = 'scores_' + str(parameters['expsuffix'])

    exps = [k for k in range(1, 8)]

    xlabel = 'number of training subjects'
    xshift = 0.4
    width = 0.15

    # Load data
    with open('{}/{}.pkl'.format(basepath, filename), 'rb') as f:
        scores = pickle.load(f)

    methods = [
        ['coroICA_base_{}'.format(EXPIDENTIFIER), 'coroICA'],
        ['fastICA', 'fastICA'],
        ['uwedgeICA_var', 'choiICA (var)'],
        ['uwedgeICA_var-TD', 'choiICA (TD\&var)'],
        ['uwedgeICA_TD', 'choiICA (TD)'],
        ['uwedgeICA_SOBI', 'SOBI'],
    ]
    basemethod = 'coroICA_base_{}'.format(EXPIDENTIFIER)
    if paired:
        methods = [m
                   for m in methods
                   if m[0] != basemethod]
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
    fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 3))

    # prepare scores
    for inout in ['insample', 'outsample']:
        for metkey in methods_keys:
            for exp in exps:
                if paired:
                    if metkey != basemethod:
                        scores[inout][metkey][exp] = np.hstack([
                            [s / g for s, g in zip(sc, gr) if not np.isnan(s)]
                            for sc, gr in zip(
                                    scores[inout][metkey][exp],
                                    scores[inout][basemethod][exp])])
                else:
                    scores[inout][metkey][exp] = np.hstack([
                        sc[~np.isnan(sc)]
                        for sc in scores[inout][metkey][exp]])

    # Prepare data
    for inout in ['insample', 'outsample']:
        if inout == 'insample':
            ax = plt.subplot(211)
            plt.title('in-sample')
        else:
            ax = plt.subplot(212)
            plt.title('out-of-sample')

        vios = []
        for metkey in methods_keys:
            nextmet = []
            for exp in exps:
                nextmet.append(scores[inout][metkey][exp])
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
            if paired:
                for x_pos, y_pos, percentage in zip(
                        pos,
                        [np.min([1., np.min(k)]) for k in vio],
                        ['{:.0f}\%'.format(
                            100 * np.mean(np.asarray(k) < 1))
                         for k in vio]):
                    plt.text(x_pos,
                             y_pos,
                             percentage,
                             rotation=45,
                             horizontalalignment='center',
                             verticalalignment='top',
                             fontsize=4)

        for vio, col in zip(violins, colors[:len(violins)]):
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                vp = vio[partname]
                vp.set_edgecolor(col)
            for vp in vio['bodies']:
                vp.set_facecolor(col)
                vp.set_edgecolor(col)

        if paired:
            ax.axhline(1, color='black', linewidth=1)
            plt.ylabel('MCIS fraction')
        else:
            plt.ylabel('MCIS')

        if inout == 'outsample':
            ax.set_xticks(np.arange(1, len(exps) + 1))
            ax.set_xticklabels(['{}'.format(exp) for exp in exps])
            plt.xlabel(xlabel)
        else:
            patches = [
                mpatches.Patch(color=vio['bodies'][0].get_facecolor()[0])
                for vio in violins]
            lgd = ax.legend(patches, methods_labels,
                            prop={'size': 6},
                            loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xticks([])
            ax.set_xticklabels([])

    plt.subplots_adjust(hspace=.4, bottom=.15)

    plt.tight_layout(pad=0.0, h_pad=2)
    fig.set_size_inches(productionplot.TEXTWIDTH, 3)
    fig.savefig('{}/{}.{}.pdf'.format(
        basepath, filename, paired),
                bbox_inches='tight',
                bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    parameters = {}
    parameters['expname'] = EXPIDENTIFIER
    parameters['expsuffix'] = 'coroICA_base_{}'.format(EXPIDENTIFIER)

    plot(parameters, paired=False)
    plot(parameters, paired=True)
