import productionplot
import numpy as np
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set_style('darkgrid')


def plot_garch(expname, exp_labels, suffix, combined=False,
               fig=None, ax=None, single_axis=False, title=None, xy=[True, True]):

    ## Load data
    res = pickle.load(open('./simulations/' + expname + '.pkl', 'rb'))

    ## Initialize plot
    readout = np.array([0, 1, 2, 4, 5])
    exp_labels = exp_labels[readout]
    if not combined:
        fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 6))
        ax = plt.subplot(111)

    xlabel = 'confounding strength'
    xshift = 0.3
    width = 0.1
    centerpos = np.arange(1, len(exp_labels)+1)
    pos0 = centerpos - xshift
    pos1 = centerpos - 2/3*xshift
    pos2 = centerpos - 1/3*xshift
    pos3 = centerpos + 1/3*xshift
    pos4 = centerpos + 2/3*xshift
    pos5 = centerpos + xshift
    pos6 = centerpos

    ## Read out data
    md_mat = np.empty((len(res[0]['result'][0]['score_MD']), len(res), len(res[0]['result'])))
    for idx in range(len(md_mat)):
        for k in range(len(res)):
            for l in range(len(res[0]['result'])):
                md_mat[idx, k, l] = res[k]['result'][l]['score_MD'][idx]

    vio_coroICA0 = [[res[strap]['result'][sig]['score_MD'][0]
                     for strap in range(len(res))]
                    for sig in readout]
    vio_coroICA1 = [[res[strap]['result'][sig]['score_MD'][1]
                     for strap in range(len(res))]
                    for sig in readout]
    vio_coroICA2 = [[res[strap]['result'][sig]['score_MD'][2]
                     for strap in range(len(res))]
                    for sig in readout]
    vio_uwedgeICA0 = [[res[strap]['result'][sig]['score_MD'][3]
                       for strap in range(len(res))]
                      for sig in readout]
    vio_uwedgeICA1 = [[res[strap]['result'][sig]['score_MD'][4]
                       for strap in range(len(res))]
                      for sig in readout]
    vio_uwedgeICA2 = [[res[strap]['result'][sig]['score_MD'][5]
                       for strap in range(len(res))]
                      for sig in readout]
    vio_fastICA = [[res[strap]['result'][sig]['score_MD'][6]
                    for strap in range(len(res))]
                   for sig in readout]

    ## Construct plot
    v_coroICA0 = ax.violinplot(vio_coroICA0, pos0,
                               points=100, widths=width,
                               showmeans=False, showextrema=True, showmedians=True)
    v_coroICA1 = ax.violinplot(vio_coroICA1, pos1,
                               points=100, widths=width,
                               showmeans=False, showextrema=True, showmedians=True)
    v_coroICA2 = ax.violinplot(vio_coroICA2, pos2,
                               points=100, widths=width,
                               showmeans=False, showextrema=True, showmedians=True)
    v_uwedgeICA0 = ax.violinplot(vio_uwedgeICA0, pos3,
                                 points=100, widths=width,
                                 showmeans=False, showextrema=True, showmedians=True)
    v_uwedgeICA1 = ax.violinplot(vio_uwedgeICA1, pos4,
                                 points=100, widths=width,
                                 showmeans=False, showextrema=True, showmedians=True)
    v_uwedgeICA2 = ax.violinplot(vio_uwedgeICA2, pos5,
                                 points=100, widths=width,
                                 showmeans=False, showextrema=True, showmedians=True)
    v_fastICA = ax.violinplot(vio_fastICA, pos6,
                              points=100, widths=width,
                              showmeans=False, showextrema=True, showmedians=True)

    # Methods
    methods = [['coroICA_var', 'coroICA (var)'],
               ['coroICA_var_TD', 'coroICA (var\&TD)'],
               ['coroICA_TD', 'coroICA (TD)'],
               ['fastICA', 'fastICA'],
               ['uwedgeICA_var', 'choiICA (var)'],
               ['uwedgeICA_var_TD', 'choiICA (var\&TD)'],
               ['uwedgeICA_TD', 'choiICA (TD)']]
    colors = productionplot.colormapping(methods)

    for vio, col in zip([v_coroICA0, v_coroICA1, v_coroICA2, v_fastICA,
                         v_uwedgeICA0, v_uwedgeICA1, v_uwedgeICA2],
                        colors):
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = vio[partname]
            vp.set_edgecolor(col)
            for vp in vio['bodies']:
                vp.set_facecolor(col)
                vp.set_edgecolor(col)

    if xy[1]:
        ax.set_ylabel("MD")
    if xy[0]:
        ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(1, len(exp_labels)+1))
    ax.set_xticklabels(exp_labels)

    if title is not None:
        ax.set_title(title)

    if not combined:
        patches = [
            mpatches.Patch(color=v_coroICA0['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_coroICA1['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_coroICA2['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_fastICA['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA0['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA1['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA2['bodies'][0].get_facecolor()[0]),
        ]
        lgd = ax.legend(patches, [m[1] for m in methods],
                        loc='center left')
    if single_axis:
        patches = [
            mpatches.Patch(color=v_coroICA0['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_coroICA1['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_coroICA2['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_fastICA['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA0['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA1['bodies'][0].get_facecolor()[0]),
            mpatches.Patch(color=v_uwedgeICA2['bodies'][0].get_facecolor()[0]),
        ]
        lgd = fig.legend(
            patches, [m[1] for m in methods],
            prop={'size': 6},
            ncol=4,
            loc='lower center')
    else:
        lgd = None

    if not combined:
        plotname = expname+'.pdf'
        plt.tight_layout(pad=0.0, h_pad=0.8)
        fig.set_size_inches(productionplot.TEXTWIDTH, 4)
        fig.savefig('./simulations/simulation_' + plotname, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
    return lgd


def plot_garch_all():

    fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 3.5))
    ax = [None]*6
    ax[0] = plt.subplot(231)
    ax[1] = plt.subplot(232)
    ax[2] = plt.subplot(233)
    ax[3] = plt.subplot(234)
    ax[4] = plt.subplot(235)
    ax[5] = plt.subplot(236)
    xy = [[False, True], [False, False], [False, False],
          [True, True], [True, False], [True, False]]
    titles = ['var signal - AR noise',
              'TD signal - AR noise',
              'var-TD signal - AR noise',
              'var signal - iid noise',
              'TD signal - iid noise',
              'var-TD signal - iid noise']

    for i in range(6):
        expname = 'experiment' + str(3+i)
        exp_labels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.6])
        if i == 4:
            lgd = plot_garch(expname, exp_labels, None, True, fig, ax[i],
                             True, titles[i], xy[i])
        else:
            plot_garch(expname, exp_labels, None, True, fig, ax[i],
                       False, titles[i], xy[i])

    plotname = 'experiment345678.pdf'
    plt.tight_layout(pad=0.4, h_pad=0.8)
    plt.subplots_adjust(bottom=0.2)
    fig.set_size_inches(productionplot.TEXTWIDTH, 3.5)
    fig.savefig('./simulations/simulation_' + plotname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def plot_ca(expname, scaled, diff, exp_labels, suffix):

    ## Set up plotting parameters
    if expname == 'experiment1':
        xlabel = 'confounding strength'
    else:
        xlabel = 'signal strength'
    xshift = 0.2
    width = 0.15

    ## Load data
    res = pickle.load(open('./simulations/' + expname + '.pkl', 'rb'))

    keep_ind = np.repeat(True, len(res))
    for k in range(len(res)):
        for l in range(len(exp_labels)):
            if type(res[k]['result'][l]) == str:
                print(res[k]['result'][l])
                keep_ind[k] = False
    res = [res[k] for k in range(len(res)) if keep_ind[k]]
    res = res[:1000]

    ## Initialize plot
    fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 4))

    ## Prepare data
    for score in ['score_MD_in', 'score_stab_in', 'score_stab_out']:
        if score == 'score_MD_in':
            ax = plt.subplot(311)
            plt.title('minimum distance index')
        elif score == 'score_stab_in':
            ax = plt.subplot(312)
            plt.title('in-sample stability')
        else:
            ax = plt.subplot(313)
            plt.title('out-of-sample stability')

        centerpos = np.arange(1, len(exp_labels)+1)
        if diff:
            pos_coroICA = centerpos - xshift
        else:
            pos_rand = centerpos - 3/2*xshift
            pos_ica = centerpos - 1/2*xshift
            pos_ica2 = centerpos + 1/2*xshift
            pos_coroICA = centerpos + 3/2*xshift

        if score == 'score_stab_in':
            vio_rand = [[res[strap]['result'][sig]['score_stab_random_in']
                         for strap in range(len(res))]
                        for sig in range(len(exp_labels))]
        elif score == 'score_stab_out':
            vio_rand = [[res[strap]['result'][sig]['score_stab_random_out']
                         for strap in range(len(res))]
                        for sig in range(len(exp_labels))]
        else:
            vio_rand = [[res[strap]['result'][sig]['rand_MD']
                         for strap in range(len(res))]
                        for sig in range(len(exp_labels))]
        vio_rand = [list(np.array(tmp).flatten())
                    for tmp in vio_rand]
        vio_ica = [[res[strap]['result'][sig]['fastICA_1'][score]
                    for strap in range(len(res))]
                   for sig in range(len(exp_labels))]
        vio_ica2 = [[res[strap]['result'][sig]['uwedgeICA_default_4'][score]
                     for strap in range(len(res))]
                    for sig in range(len(exp_labels))]
        vio_coroICA = [[res[strap]['result'][sig]['coroICA_0'][score]
                        for strap in range(len(res))]
                       for sig in range(len(exp_labels))]

        # scale data
        if scaled:
            mu = [np.copy(evio.mean(axis=1)) for evio in vio_rand]
            std = [np.copy(evio.std(axis=1)) for evio in vio_rand]
            vio_rand = [((mu[k].reshape(-1, 1)-vio_rand[k])/std[k].reshape(-1, 1)).flatten()
                        for k in range(1)]
            vio_ica = [(mu[k]-vio_ica[k])/std[k] for k in range(1)]
            vio_ica2 = [(mu[k]-vio_ica2[k])/std[k] for k in range(1)]
            vio_coroICA = [(mu[k]-vio_coroICA[k])/std[k] for k in range(1)]

        ## Construct plot
        if not diff:
            v_rand = ax.violinplot(vio_rand, pos_rand,
                                   points=100, widths=width,
                                   showmeans=False, showextrema=True, showmedians=True)
            v_ica = ax.violinplot(vio_ica, pos_ica,
                                  points=100, widths=width,
                                  showmeans=False, showextrema=True, showmedians=True)
            v_ica2 = ax.violinplot(vio_ica2, pos_ica2,
                                   points=100, widths=width,
                                   showmeans=False, showextrema=True, showmedians=True)
        v_coroICA = ax.violinplot(vio_coroICA, pos_coroICA,
                                  points=100, widths=width,
                                  showmeans=False, showextrema=True, showmedians=True)

        # Methods
        methods = [['coroICA_var', 'coroICA'],
                   ['fastICA', 'fastICA'],
                   ['uwedgeICA_var', 'choiICA (var)'],
                   ['random', 'random']]
        colors = productionplot.colormapping(methods)

        for vio, col in zip([v_coroICA, v_ica, v_ica2, v_rand],
                            colors):
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                vp = vio[partname]
                vp.set_edgecolor(col)
            for vp in vio['bodies']:
                vp.set_facecolor(col)
                vp.set_edgecolor(col)

        plt.ylabel(score)
        if scaled:
            ax.set_ylabel('standard deviations')
        else:
            if score == 'score_MD_in':
                ax.set_ylabel('MD')
            else:
                ax.set_ylabel('MCIS')
        if score == 'score_stab_out':
            ax.set_xticks(np.arange(1, len(exp_labels)+1))
            ax.set_xticklabels(exp_labels)
            plt.xlabel(xlabel)
            if diff:
                patches = [
                    mpatches.Patch(color=v_coroICA['bodies'][0].get_facecolor()[0])]
                lgd = ax.legend(patches, ['coroICA'], prop={'size': 6},
                                loc='upper right', bbox_to_anchor=(1.15, 1.3585))
        elif score == 'score_MD_in':
            patches = [
                mpatches.Patch(color=v_coroICA['bodies'][0].get_facecolor()[0]),
                mpatches.Patch(color=v_ica['bodies'][0].get_facecolor()[0]),
                mpatches.Patch(color=v_ica2['bodies'][0].get_facecolor()[0]),
                mpatches.Patch(color=v_rand['bodies'][0].get_facecolor()[0]),
            ]
            lgd = ax.legend(patches, [m[1] for m in methods],
                            prop={'size': 6},
                            loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xticks(np.arange(1, len(exp_labels)+1))
            ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(1, len(exp_labels)+1))
            ax.set_xticklabels([])

    plt.subplots_adjust(hspace=.4, bottom=.15)

    # save figure
    scaled = 'unscaled'
    diff = 'absolute'

    plotname = expname+'_'+scaled+'_'+diff+'_instability.pdf'
    plt.tight_layout(pad=0.0, h_pad=0.8)
    fig.set_size_inches(productionplot.TEXTWIDTH, 4)
    fig.savefig('./simulations/simulation_' + plotname, bbox_inches='tight',
                bbox_extra_artists=(lgd,))


# Main function - Runs all experiments
if __name__ == '__main__':

    try:
        expname = sys.argv[1]
    except IndexError:
        print('Please provide argument experiment1, experiment2, ... '
              'to select which experiment to run.')
        exit()

    if expname == 'experiment1':
        # Signal-strength simulation
        scaled = False
        diff = False
        subenvs = [0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]
        exp_labels = subenvs
        plot_ca(expname, scaled, diff, exp_labels, None)
    elif expname == 'experiment2':
        # zero confounding (robustness) simulation
        scaled = False
        diff = False
        subenvs = [0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.6]
        exp_labels = subenvs
        plot_ca(expname, scaled, diff, exp_labels, None)
    elif expname == 'experiment3':
        # GARCH 1
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment4':
        # GARCH 2
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment5':
        # GARCH 3
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment6':
        # GARCH 1 (iid noise)
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment7':
        # GARCH 2 (iid noise)
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment8':
        # GARCH 3 (iid noise)
        exp_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        plot_garch(expname, exp_labels, None)
    elif expname == 'experiment345678':
        # All GARCH
        plot_garch_all()
