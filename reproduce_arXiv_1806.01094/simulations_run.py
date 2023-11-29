import numpy as np
import pickle
import sys
from itertools import combinations
from joblib import Parallel, delayed
import sklearn
from sklearn.decomposition import FastICA
from coroica import CoroICA, UwedgeICA
import os
from utils import MDindex


## generate data from GARCH model
def generate_garch(n, burnin=100, fixed_var=False, time_dep=True):
    N = n + burnin
    # sample variance process
    x = np.zeros(N)+1
    varvec = np.zeros(N)+1
    avec = np.array([0.005, 0.026, 0.97])
    bad_res = True
    while bad_res:
        noise = np.random.normal(0, 1, N)
        p = np.random.randint(1, 10, 1)
        bvec = np.random.normal(0, 1, p)/np.arange(2, p+2)**2
        for i in range(1, N):
            if fixed_var:
                varvec[i] = 1
            else:
                varvec[i] = (avec[0] + avec[1]*x[i-1]**2 + avec[2]*varvec[i-1])
            if time_dep:
                x[i] = np.sum(bvec*x[np.arange(i-1, i-p-1, -1)]) + np.sqrt(varvec[i])*noise[i]
            else:
                x[i] = np.sqrt(varvec[i])*noise[i]
        if np.sum(np.isnan(x)) > 0:
            bad_res = True
        else:
            bad_res = (np.sum(np.abs(varvec) < 30) != np.size(varvec) or
                       np.sum(np.abs(x) < 50) != np.size(x))
    x = x/np.sqrt(np.var(x))
    return x[burnin:], varvec[burnin:]


# Function to simulate block-wise shifted variance data
def simulate_blockwise(dim=16,
                       confdim=16,
                       no_envs=5,
                       no_subenvs=10,
                       no_samples=30000,
                       signal_strength=1,
                       conf_strength=1):
    A = np.random.randn(dim, dim)

    S = np.zeros((dim, no_envs*no_samples))
    env_index = np.repeat(range(no_envs), no_samples).reshape(1, -1)
    subenv_index = randsubgroup(env_index, int(no_samples/no_subenvs))

    afit = 0.1
    bfit = 3*signal_strength + afit
    cfit = 2*conf_strength - afit

    C = np.random.randn(dim, confdim)/np.sqrt(confdim)
    for env in range(no_envs):
        Senv = np.empty((dim, no_samples))
        subenv_sizes = np.unique(subenv_index[0, env_index[0, :] == env],
                                 return_counts=True)[1]
        for k in range(dim):
            sigmas_signal = np.sqrt(np.random.uniform(afit, bfit, len(subenv_sizes)))
            Senv[k, :] = np.random.normal(0,
                                          np.repeat(sigmas_signal, subenv_sizes),
                                          no_samples)
        if conf_strength != 0:
            Henv = np.empty((confdim, no_samples))
            for k in range(confdim):
                sigma_conf = np.sqrt(np.random.uniform(afit, cfit, 1))
                Henv[k, :] = np.random.normal(0,
                                              sigma_conf,
                                              no_samples)
        else:
            Henv = np.zeros((confdim, no_samples))
        S[:, env_index[0, :] == env] = Senv + np.dot(C, Henv)

    X = A.dot(S)

    return {'A': A, 'C': C, 'S': S, 'X': X,
            'env_index': env_index, 'subenv_index': subenv_index}


def simulate_GARCH(dim=16,
                   confdim=16,
                   no_envs=1,
                   no_subenvs=10,
                   no_samples=30000,
                   signal_strength=1,
                   conf_strength=1,
                   experiment_type='varsig',
                   ar_noise=True):
    if experiment_type == 'varsig':
        fixed_var = False
        time_dep = False
    elif experiment_type == 'varsig_TD':
        fixed_var = False
        time_dep = True
    elif experiment_type == 'TD':
        fixed_var = True
        time_dep = True

    A = np.random.randn(dim, dim)
    C = np.random.randn(dim, confdim)/np.sqrt(confdim)

    S = np.zeros((dim, no_envs*no_samples))
    env_index = np.repeat(range(no_envs), no_samples).reshape(1, -1)
    subenv_index = randsubgroup(env_index, int(no_samples/no_subenvs))

    for env in range(no_envs):
        # Signal part
        Senv = np.empty((dim, no_samples))
        subenvs = np.unique(subenv_index[0, env_index[0, :] == env],
                            return_counts=True)
        for k in range(dim):
            for idx, sub in enumerate(subenvs[0]):
                subenv_ind = subenv_index[0, env_index[0, :] == env] == sub
                Senv[k, subenv_ind] = generate_garch(subenvs[1][idx],
                                                     fixed_var=fixed_var, time_dep=time_dep)[0]
        # Noise part
        Henv = np.zeros((confdim, no_samples))
        if conf_strength != 0:
            for k in range(confdim):
                if ar_noise:
                    Henv[k, :] = generate_garch(no_samples,
                                                fixed_var=True, time_dep=True)[0]*np.sqrt(
                                                    conf_strength)
                else:
                    Henv[k, :] = generate_garch(no_samples,
                                                fixed_var=True, time_dep=False)[0]*np.sqrt(
                                                    conf_strength)
        # Combine signal and noise
        S[:, env_index[0, :] == env] = Senv + np.dot(C, Henv)

    X = A.dot(S)

    return {'A': A, 'C': C, 'S': S, 'X': X,
            'env_index': env_index, 'subenv_index': subenv_index}


# Funtion to generate random subgroups
def randsubgroup(env, nosamples):
    subenv = np.zeros(env.shape)
    for e in np.unique(env):
        subenv[np.where(env == e)] = randgroup((env == e).sum(),
                                               nosamples) + subenv.max() + 1
    return subenv


# Function to generate random groups
def randgroup(length, nosamples):
    envs = int(np.round(length/nosamples))
    changepoints = [int(np.round(a))
                    for a in np.linspace(0, length, envs+1)]
    changepoints = list(set(changepoints))
    changepoints.sort()
    width = changepoints[1]
    changepoints = [0] + [int(np.round(a + np.random.randn()*np.sqrt(width/3)))
                          for a in changepoints[1:-1]] + [changepoints[-1]]
    index = np.zeros(length)
    for (i, a, b) in zip(range(envs), changepoints[:-1], changepoints[1:]):
        index[a:b] = i
    return index.reshape(1, -1)


# Function for generating rigid subenv
def rigidsubenv(env, nosamples):
    subenv = np.zeros(env.shape)
    for e in np.unique(env):
        subenv[np.where(env == e)] = rigidenv((env == e).sum(),
                                              nosamples) + subenv.max() + 1
    return subenv


# Function for generating rigid env
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


# Helper function for MCIS
def inner_score(V, groupcovmats, subenvdiffmats):
    # scale V for numerical reasons, does not change score
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    num_groups = len(groupcovmats)
    scoremat = np.zeros(V.shape[0])
    for g in range(num_groups):
        scalemat = np.sqrt(np.abs(np.diag(V.dot(groupcovmats[g].dot(V.T)))))
        scalemat = np.outer(scalemat, scalemat)
        differences = np.sum([((V.dot(m.dot(V.T))) / scalemat)**2
                              for m in subenvdiffmats[g]], axis=0)
        scoremat = scoremat + 1 / (len(subenvdiffmats[g])) * differences
    scoremat = scoremat / num_groups
    return np.sqrt(np.mean(scoremat[~np.eye(scoremat.shape[0], dtype=bool)]))


# Function to compute MCIS
def MCIS(X, V, env, subenv):
    # precompute all covariance matrices of X
    groupcovmats = []
    subenvcovmats = []
    for group in np.unique(env):
        X_group = X[:, env == group]
        subenv_group = subenv[env == group]
        sublist = []
        # precompute all subgroupcovmats
        for subgroup in np.unique(subenv_group):
            sublist.append(np.cov(X_group[:, subenv_group == subgroup]))
        # collect subgroupcovmats
        subenvcovmats.append(np.copy(sublist))
        # precompute groupcovmat
        groupcovmats.append(np.cov(X_group))

    subenvdiffmats = []
    for g in range(len(groupcovmats)):
        newlist = []
        for e, f in combinations(subenvcovmats[g], 2):
            newlist.append((e-f))
        subenvdiffmats.append(np.copy(newlist))

    # compute scores
    mcis = inner_score(V, groupcovmats, subenvdiffmats)

    return mcis


# Function to compute MCIS for random projections
def random_MCIS(X, env, subenv, B=100):
    # precompute all covariance matrices of X
    groupcovmats = []
    subenvcovmats = []
    for group in np.unique(env):
        X_group = X[:, env == group]
        subenv_group = subenv[env == group]
        sublist = []
        # precompute all subgroupcovmats
        for subgroup in np.unique(subenv_group):
            sublist.append(np.cov(X_group[:, subenv_group == subgroup]))
        # collect subgroupcovmats
        subenvcovmats.append(np.copy(sublist))
        # precompute groupcovmat
        groupcovmats.append(np.cov(X_group))

    subenvdiffmats = []
    for g in range(len(groupcovmats)):
        newlist = []
        for e, f in combinations(subenvcovmats[g], 2):
            newlist.append((e-f))
        subenvdiffmats.append(np.copy(newlist))

    # compute resampled scores
    boot_mcis = np.empty(B)
    for i in range(B):
        Vboot = np.random.randn(X.shape[0], X.shape[0])
        boot_mcis[i] = inner_score(Vboot, groupcovmats, subenvdiffmats)

    return boot_mcis


# Function to compute CA score
def CAscore(V1, V2):
    d = V1.shape[0]
    bestcorrs = np.empty(d)
    corrmat = np.abs(np.corrcoef(V1, V2)[0:d, d:2*d])
    for k in range(d):
        i, j = np.unravel_index(corrmat.argmax(), corrmat.shape)
        bestcorrs[k] = corrmat[i, j]
        corrmat[i, :] = 0
        corrmat[:, j] = 0
    return np.mean(bestcorrs)


# Function to perform scoring on single identifier
def splitICA_simulation(identifier, allsamples, allsubject_index,
                        trueA, ICAs, save=True):

    # convert identifier to np.array
    if not type(identifier) == dict:
        raise RuntimeError('no split identifiers dictionary given')
    else:
        trainset = np.isin(allsubject_index[0, :], identifier['train'])
        testset = np.isin(allsubject_index[0, :], identifier['test'])

    fname = '{}_train_{}.pkl'.format(
        '_'.join([str(k) for k in identifier['test']]),
        '_'.join([str(k) for k in identifier['train']]))

    # if file was already computed before
    if os.path.isfile(fname) and save:
        with open(fname, 'rb') as f:
            res = pickle.load(f)
    else:
        eps = 0
        n_iter_max = 10000

        # iterate over ICAs
        V = [None]*2
        Shat_in = [None]*2
        Shat_out = [None]*2
        numICAs = len(ICAs)
        sets = [trainset, testset]
        for xx, fitset in enumerate(sets):
            V[xx] = []
            Shat_in[xx] = []
            Shat_out[xx] = []
            for idx in range(numICAs):
                if ICAs[idx][0] == 'coroICA':
                    ica = CoroICA(n_components=None,
                                  partitionsize=ICAs[idx][1][0],
                                  pairing=ICAs[idx][1][1],
                                  max_iter=n_iter_max,
                                  tol=eps)
                    ica.fit(allsamples[:, fitset].T,
                            group_index=allsubject_index[0, fitset])
                    Vtmp = ica.V_
                elif ICAs[idx][0] == 'fastICA':
                    ica = FastICA(random_state=0)
                    ica.fit(allsamples[:, fitset].T)
                    Vtmp = ica.components_
                elif 'coroICA_' in ICAs[idx][0]:
                    ica = sklearn.clone(ICAs[idx][1])
                    ica.fit(allsamples[:, fitset].T,
                            group_index=allsubject_index[0, fitset])
                    Vtmp = ica.V_
                elif 'uwedgeICA_' in ICAs[idx][0]:
                    ica = sklearn.clone(ICAs[idx][1])
                    ica.fit(allsamples[:, fitset].T)
                    Vtmp = ica.V_
                else:
                    raise RuntimeError('ICA method "{}" unknown'.format(ICAs[idx][0]))
                V[xx].append(np.copy(Vtmp))
                Shat_in[xx].append(Vtmp.dot(allsamples[:, fitset]))
                Shat_out[xx].append(Vtmp.dot(allsamples[:, ~fitset]))
        trueV = np.linalg.inv(trueA)
        # generate subgroups
        train_partition = rigidsubenv(allsubject_index[:, trainset], ICAs[0][1][0])
        test_partition = rigidsubenv(allsubject_index[:, testset], ICAs[0][1][0])
        # compute random in- and out-of-sample scores
        rand_in = random_MCIS(allsamples[:, trainset],
                              allsubject_index[0, trainset],
                              train_partition[0, :])
        rand_out = random_MCIS(allsamples[:, testset],
                               allsubject_index[0, testset],
                               test_partition[0, :])
        rand_CA = [CAscore(np.random.randn(*trueV.shape), trueV)
                   for i in range(100)]
        rand_MD = [MDindex(trueA, np.random.randn(*trueV.shape))
                   for i in range(100)]

        # compute scores in- and out-of-sample
        score_CA_in = np.empty(numICAs)
        score_CA_out = np.empty(numICAs)
        score_MD_in = np.empty(numICAs)
        score_MD_out = np.empty(numICAs)
        score_stab_in = np.empty(numICAs)
        score_stab_out = np.empty(numICAs)
        for idx in range(numICAs):
            score_CA_in[idx] = CAscore(V[0][idx], trueV)
            score_CA_out[idx] = CAscore(V[1][idx], trueV)
            score_MD_in[idx] = MDindex(trueA, V[0][idx])
            score_MD_out[idx] = MDindex(trueA, V[1][idx])
            score_stab_in[idx] = MCIS(allsamples[:, sets[0]], V[0][idx],
                                      allsubject_index[0, sets[0]],
                                      train_partition[0, :])
            score_stab_out[idx] = MCIS(allsamples[:, sets[1]], V[0][idx],
                                       allsubject_index[0, sets[1]],
                                       test_partition[0, :])

        res = {'identifier': identifier,
               'rand_CA': rand_CA,
               'rand_MD': rand_MD,
               'score_stab_random_in': rand_in,
               'score_stab_random_out': rand_out}
        print(score_MD_in)

        for idx in range(numICAs):
            res['{}_{}'.format(ICAs[idx][0], idx)] = {
                'method': ICAs[idx],
                'Vtrain': V[0][idx],
                'Vtest': V[1][idx],
                'score_CA_in': score_CA_in[idx],
                'score_CA_out': score_CA_out[idx],
                'score_MD_in': score_MD_in[idx],
                'score_MD_out': score_MD_out[idx],
                'score_stab_in': score_stab_in[idx],
                'score_stab_out': score_stab_out[idx]}
        if save:
            with open('./simulations/' + fname, 'wb') as f:
                pickle.dump(res, f)
    return res


# Function to perform a single iteration of simulation
def resample_splits(iteration, ICAs, dim, confdim,
                    no_envs, no_subenvs,
                    no_samples,
                    experiment_type,
                    signal_strength,
                    conf_strength):

    print("Starting iteration " + str(iteration))

    # iterate over all variance_conf/variance_signal settings
    len_ss1 = len(conf_strength)
    len_ss2 = len(signal_strength)
    strengths = np.empty((len_ss1*len_ss2, 2))
    strengths[:, 0] = np.repeat(conf_strength, len_ss2)
    strengths[:, 1] = np.tile(signal_strength, len_ss1)
    total_results = [None]*len_ss1*len_ss2
    for i in range(len_ss1*len_ss2):
        # status output
        print("Starting computation for signal strength " +
              str(strengths[i, :]))

        # simulate data for given signal_strength
        np.random.seed(iteration+i*100000)
        if experiment_type == "blockwise":
            data = simulate_blockwise(dim, confdim,
                                      no_envs, no_subenvs,
                                      no_samples,
                                      strengths[i, 1],
                                      strengths[i, 0])
        elif experiment_type == "GARCH":
            data = simulate_GARCH(dim, confdim,
                                  no_envs, no_subenvs,
                                  no_samples,
                                  strengths[i, 1],
                                  strengths[i, 0])
        X = data['X']
        trueA = data['A']
        env_index = data['env_index']

        # create identifier
        trainenvs = np.arange(0, int(np.floor(no_envs/2)), dtype='int')
        testenvs = np.arange(int(np.floor(no_envs/2)), no_envs, dtype='int')
        identifier = {'train': trainenvs, 'test': testenvs}
        # compute ICA/coroICA with leave-one-out
        try:
            results = splitICA_simulation(identifier, X, env_index, trueA,
                                          ICAs, save=True)
        except ValueError:
            print("Encountered error in fastICA:")
            print(iteration+i*100000)
            results = 'fastICA_error'

        total_results[i] = results

    result = {'result': total_results,
              'confdim': confdim,
              'no_envs': no_envs,
              'no_subenvs': no_subenvs,
              'no_samples': no_samples,
              'signal_strength': signal_strength,
              'conf_strength': conf_strength}

    return result


# Function to run simulation experiment
def run_split(parameters):

    # read out parameters
    no_envs = parameters['no_envs']
    no_subenvs = parameters['no_subenvs']
    no_samples = parameters['no_samples']
    dim = parameters['dim']
    confdim = parameters['confdim']
    signal_strength = parameters['signal_strength']
    conf_strength = parameters['conf_strength']
    B = parameters['B']
    n_jobs = parameters['n_jobs']
    ICAs = parameters['ICAs']
    experiment_type = parameters['experiment_type']
    expname = parameters['expname']

    # run multiple runs of simulation
    resample_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(resample_splits, check_pickle=False)(
            iteration, ICAs, dim, confdim, no_envs, no_subenvs,
            no_samples, experiment_type, signal_strength, conf_strength)
        for iteration in list(range(B)))

    # save data
    with open('./simulations/' + expname + '.pkl', 'wb') as f:
        pickle.dump(resample_results, f)

    return "Done"


# Function to perform a single iteration of simulation
def resample_mdanalysis(iteration, ICAs, dim, confdim,
                        no_envs, no_subenvs,
                        no_samples,
                        partitionsize,
                        experiment_type,
                        ar_noise,
                        signal_strength,
                        conf_strength):

    print("Starting iteration " + str(iteration))

    # iterate over all variance_conf/variance_signal settings
    len_ss1 = len(conf_strength)
    len_ss2 = len(signal_strength)
    strengths = np.empty((len_ss1*len_ss2, 2))
    strengths[:, 0] = np.repeat(conf_strength, len_ss2)
    strengths[:, 1] = np.tile(signal_strength, len_ss1)
    total_results = [None]*len_ss1*len_ss2
    for i in range(len_ss1*len_ss2):
        # status output
        print("Starting computation for signal strength " +
              str(strengths[i, :]))

        # simulate data for given signal_strength
        np.random.seed(iteration+i*100000)
        if experiment_type == "blockwise":
            data = simulate_blockwise(dim, confdim,
                                      no_envs, no_subenvs,
                                      no_samples,
                                      strengths[i, 1],
                                      strengths[i, 0])
        else:
            data = simulate_GARCH(dim, confdim,
                                  no_envs, no_subenvs,
                                  no_samples,
                                  strengths[i, 1],
                                  strengths[i, 0],
                                  experiment_type,
                                  ar_noise)
        print("Generated data. Applying ICA and scoring...")
        X = data['X']
        trueA = data['A']
        env_index = data['env_index']
        numICAs = len(ICAs)

        # compute ICA/coroICA with leave-one-out
        score_MD = np.empty(numICAs)
        Vhat = [None]*numICAs
        for idx in range(numICAs):
            if ICAs[idx][0] == 'coroICA':
                ica = CoroICA(n_components=None,
                              partitionsize=ICAs[idx][1][0],
                              pairing=ICAs[idx][1][1],
                              max_iter=1000,
                              tol=0)
                ica.fit(X.T,
                        group_index=env_index[0, :])
                Vhat[idx] = ica.V_
            elif ICAs[idx][0] == 'fastICA':
                ica = FastICA(random_state=0)
                ica.fit(X.T)
                Vhat[idx] = ica.components_
            elif 'coroICA_' in ICAs[idx][0]:
                ica = sklearn.clone(ICAs[idx][1])
                ica.fit(X.T,
                        group_index=env_index[0, :])
                Vhat[idx] = ica.V_
            elif 'uwedgeICA_' in ICAs[idx][0]:
                ica = sklearn.clone(ICAs[idx][1])
                ica.fit(X.T)
                Vhat[idx] = ica.V_
            else:
                raise RuntimeError('ICA method "{}" unknown'.format(ICAs[idx][0]))
            score_MD[idx] = MDindex(trueA, Vhat[idx])
        rand_MD = [MDindex(trueA, np.random.randn(*trueA.shape))
                   for i in range(100)]

        results = {'score_MD': score_MD,
                   'rand_MD': rand_MD,
                   'trueA': trueA,
                   'Vhat': Vhat}
        total_results[i] = results

    result = {'result': total_results,
              'confdim': confdim,
              'no_envs': no_envs,
              'no_subenvs': no_subenvs,
              'no_samples': no_samples,
              'signal_strength': signal_strength,
              'conf_strength': conf_strength}

    return result


# Function to run simulation experiment
def run_mdanalysis(parameters):

    # read out parameters
    no_envs = parameters['no_envs']
    no_subenvs = parameters['no_subenvs']
    no_samples = parameters['no_samples']
    dim = parameters['dim']
    confdim = parameters['confdim']
    partitionsize = parameters['partitionsize']
    signal_strength = parameters['signal_strength']
    conf_strength = parameters['conf_strength']
    B = parameters['B']
    n_jobs = parameters['n_jobs']
    ICAs = parameters['ICAs']
    experiment_type = parameters['experiment_type']
    ar_noise = parameters['ar_noise']
    expname = parameters['expname']

    # run multiple runs of simulation
    resample_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(resample_mdanalysis, check_pickle=False)(
            iteration, ICAs, dim, confdim, no_envs, no_subenvs,
            no_samples, partitionsize, experiment_type, ar_noise,
            signal_strength, conf_strength)
        for iteration in list(range(B)))

    # save data
    with open('./simulations/' + expname + '.pkl', 'wb') as f:
        pickle.dump(resample_results, f)

    return "Done"


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
        conf_strength = [0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 10
        parameters['no_subenvs'] = 10
        parameters['no_samples'] = 10000
        parameters['experiment_type'] = 'blockwise'
        parameters['ICAs'] = (
            ('coroICA', (int(parameters['no_samples']/10), 'complement')),
            ('fastICA', (None,)),
            ('coroICA_linear',
             CoroICA(partitionsize=int(parameters['no_samples']/10),
                     pairing='allpairs',
                     max_matrices='no_partitions',
                     tol=0, max_iter=1000)),
            ('coroICA_0.2allpairs',
             CoroICA(partitionsize=int(parameters['no_samples']/10),
                     pairing='allpairs',
                     max_matrices=1,
                     tol=0, max_iter=1000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=1000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=True,
                       timelags=[1, 5, 10],
                       tol=0, max_iter=1000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=False,
                       timelags=[1, 5, 10],
                       tol=0, max_iter=1000))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1100
        parameters['expname'] = expname
        run_split(parameters)
    elif expname == 'experiment2':
        # zero confounding (robustness) simulation
        conf_strength = [0]
        signal_strength = [0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.6, 3.2, 6.4]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 10
        parameters['no_subenvs'] = 10
        parameters['no_samples'] = 2000
        parameters['experiment_type'] = 'blockwise'
        parameters['ICAs'] = (
            ('coroICA', (int(parameters['no_samples']/10), 'complement')),
            ('fastICA', (None,)),
            ('coroICA_linear',
             CoroICA(partitionsize=int(parameters['no_samples']/10),
                     pairing='allpairs',
                     max_matrices='no_partitions',
                     tol=0, max_iter=1000)),
            ('coroICA_1allpairs',
             CoroICA(partitionsize=int(parameters['no_samples']/10),
                     pairing='allpairs',
                     max_matrices=1,
                     tol=0, max_iter=1000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=1000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=True,
                       timelags=[1, 5, 10],
                       tol=0, max_iter=1000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/10),
                       instantcov=False,
                       timelags=[1, 5, 10],
                       tol=0, max_iter=1000))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1100
        parameters['expname'] = expname
        run_split(parameters)
    elif expname == 'experiment3':
        # GARCH model (var signal) - AR noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 1
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'varsig'
        parameters['ar_noise'] = True
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
    elif expname == 'experiment4':
        # GARCH model (TD signal) - AR noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 100
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'TD'
        parameters['ar_noise'] = True
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
    elif expname == 'experiment5':
        # GARCH model (TD & var signal) - AR noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 100
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'varsig_TD'
        parameters['ar_noise'] = True
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
    elif expname == 'experiment6':
        # GARCH model (var signal) - independent noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 1
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'varsig'
        parameters['ar_noise'] = False
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
    elif expname == 'experiment7':
        # GARCH model (TD signal) - independent noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 100
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'TD'
        parameters['ar_noise'] = False
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
    elif expname == 'experiment8':
        # GARCH model (TD & var signal) - independent noise
        conf_strength = [0, 0.2, 0.4, 0.6, 0.8, 1.6]
        signal_strength = [1]
        parameters = {}
        parameters['n_jobs'] = -2
        parameters['no_envs'] = 1
        parameters['no_subenvs'] = 100
        parameters['no_samples'] = 200000
        parameters['partitionsize'] = int(parameters['no_samples']/200)
        parameters['experiment_type'] = 'varsig_TD'
        parameters['ar_noise'] = False
        parameters['ICAs'] = (
            ('coroICA_complement',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     tol=0, max_iter=2000)),
            ('coroICA_complement_lags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     tol=0, max_iter=2000)),
            ('coroICA_complement_onlylags',
             CoroICA(partitionsize=int(parameters['no_samples']/200),
                     pairing='complement',
                     timelags=[1, 2, 3],
                     instantcov=False,
                     tol=0, max_iter=2000)),
            ('uwedgeICA_default',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=None,
                       tol=0, max_iter=2000)),
            ('uwedgeICA_lags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=True,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('uwedgeICA_onlylags',
             UwedgeICA(partitionsize=int(parameters['no_samples']/200),
                       instantcov=False,
                       timelags=[1, 2, 3],
                       tol=0, max_iter=2000)),
            ('fastICA', (None,))
        )
        parameters['dim'] = 22
        parameters['signal_strength'] = signal_strength
        parameters['conf_strength'] = conf_strength
        parameters['confdim'] = 22
        parameters['B'] = 1000
        parameters['expname'] = expname
        run_mdanalysis(parameters)
