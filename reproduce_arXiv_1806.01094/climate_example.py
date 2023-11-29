import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import productionplot
from coroica import CoroICA, UwedgeICA
from sklearn.decomposition import FastICA
from utils import svarICA
from scipy.interpolate import interp1d
import seaborn as sns


# Load and plot data
# download and preprocess the data and specify basepath below

## CO2 data can be found here:
# http://ncdc.noaa.gov/paleo/study/17975
# or directly downloaded here:
# ftp://ftp.ncdc.noaa.gov/pub/data/paleo/icecore/antarctica/antarctica2015co2.xls

## Temperature data can be found here:
# https://doi.pangaea.de/10.1594/PANGAEA.683655

## Preprocess as follows:
# For both temperature and CO2 create new csv files called
# temp_clean.csv and co2_clean.csv, respectively, by only saving the
# age and CO2/Temp measurements

# Read in data
basepath = '../../data/tempco/'
TempRaw = np.asarray(pd.read_csv('{}temp_clean.csv'.format(basepath)))
CO2Raw = np.asarray(pd.read_csv('{}co2_clean.csv'.format(basepath)))
CO2Raw[:, 0] = CO2Raw[:, 0] / 1000
intTemp = interp1d(TempRaw[:, 0], TempRaw[:, 1], kind='linear')
intCO2 = interp1d(CO2Raw[:, 0], CO2Raw[:, 1], kind='linear')
Temp = intTemp(np.arange(1, 801, 0.5))
CO2 = intCO2(np.arange(1, 801, 0.5))
X = np.empty((len(Temp), 2))
X[:, 0] = np.log(CO2)
X[:, 1] = Temp


# Function that computes climate sensitivity based on causal SVAR model
def climate_sensitivity(ICA, p):
    # Select ICA
    if ICA == 'CoroICA':
        ica = svarICA(ICA=CoroICA(partitionsize=list(np.arange(50, 100, 5)),
                                  pairing='allpairs', max_matrices=1),
                      p=p)
        ica.fit(np.copy(X))
    elif ICA == 'ChoiICA (var)':
        ica = svarICA(ICA=UwedgeICA(partitionsize=list(np.arange(50, 100, 5))),
                      p=p)
        ica.fit(np.copy(X))
    elif ICA == 'FastICA':
        ica = svarICA(ICA=FastICA(),
                      p=p)
        ica.fit(np.copy(X))
    Bica1 = np.eye(2) - np.diag(1 / np.diagonal(ica.V_)).dot(ica.V_)
    Bica2 = np.eye(2) - np.diag(1 / np.diagonal(ica.V_[[1, 0], :])).dot(ica.V_[[1, 0], :])
    if np.abs(np.linalg.det(Bica1)) > 1:
        Bica = Bica2
    else:
        Bica = Bica1
    cs = Bica[1, 0] * np.log(2)
    carbon_sense = Bica[0, 1]
    return cs, carbon_sense


# Run experiment
psvec = np.arange(3, 31, 1)
cs_vec1 = np.empty((len(psvec), 2))
cs_vec2 = np.empty((len(psvec), 2))
cs_vec3 = np.empty((len(psvec), 2))
for i, ps in enumerate(psvec):
    cs_vec1[i, :] = climate_sensitivity('CoroICA', ps)
    cs_vec2[i, :] = climate_sensitivity('ChoiICA (var)', ps)
    cs_vec3[i, :] = climate_sensitivity('FastICA', ps)

ind1 = abs(cs_vec1[:, 0]) < 15
ind2 = abs(cs_vec2)[:, 0] < 15
ind3 = abs(cs_vec3[:, 0]) < 15

###
# Plot results
###

shifter = lambda x: 3 * (x - psvec[ind3].max())

fig = plt.figure(dpi=600, figsize=(0.575 * productionplot.TEXTWIDTH, 3))
p3 = plt.plot(shifter(psvec[ind3]), cs_vec3[ind3, 0], '--.', label='FastICA')
p2 = plt.plot(shifter(psvec[ind2]), cs_vec2[ind2, 0], '--.', label='ChoiICA (var)')
p1 = plt.plot(shifter(psvec[ind1]), cs_vec1[ind1, 0], '--.', label='CoroICA')
sns.distplot(cs_vec3[ind3, 0], color=p3[0].get_color(),
             hist=True, norm_hist=False, axlabel=False,
             kde=False, vertical=True)
sns.distplot(cs_vec2[ind2, 0], color=p2[0].get_color(),
             hist=True, norm_hist=False, axlabel=False,
             kde=False, vertical=True)
sns.distplot(cs_vec1[ind1, 0], color=p1[0].get_color(),
             hist=True, norm_hist=False, axlabel=False,
             kde=False, vertical=True)
plt.xlabel('number of lags')
plt.ylabel('climate sensitivity')
ax = plt.gca()
ax.axvline(shifter(psvec[ind3].max()), color='k', linewidth=1)
xticks = np.arange(3, 35, 5)
xticklabels = np.hstack([xticks[:-1], ['histogram']])
xticks = np.hstack(shifter(xticks))
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
axhxmax = (-plt.gca().get_xlim()[0] + shifter(30)) / \
    np.sum(np.abs(plt.gca().get_xlim()))
plt.axhspan(1.5, 4.5, -10000, xmax=axhxmax,
            facecolor='gray', alpha=0.4)
plt.axhspan(1, 6, -10000, xmax=axhxmax,
            facecolor='gray', alpha=0.2)
plt.axhspan(1.5, 4.5, xmin=axhxmax,
            facecolor='gray', alpha=0.1)
plt.axhspan(1, 6, xmin=axhxmax,
            facecolor='gray', alpha=0.05)
lh = [p1[0], p2[0], p3[0]]
lgd = plt.legend(lh,
                 [l.get_label() for l in lh],
                 loc='lower center',
                 bbox_to_anchor=(0.5, 1.),
                 ncol=3,
                 prop={'size': 6})
plotname = 'climate_sensitivity.pdf'
plt.tight_layout(pad=0.0, h_pad=0.5)
fig.set_size_inches(0.575 * productionplot.TEXTWIDTH, 2.6)
fig.savefig(plotname, bbox_inches='tight', bbox_extra_artists=(lgd,))

###
# Plot raw data
###

fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH * 0.375, 2.6))
ax0 = plt.subplot(211)
ax0.plot(X[:, 0], marker='o', linewidth=0.3, markersize=0.4, alpha=0.9)
plt.xticks([])
plt.ylabel('$\log($CO$_2)$')
ax1 = plt.subplot(212)
ax1.plot(np.arange(len(X[:, 1])) / 2, X[:, 1],
         marker='o', linewidth=0.3, markersize=0.4, alpha=0.9)
plt.xlabel('ky B.P. (kilo-years before present)')
plt.ylabel('$\delta T$')
plotname = 'raw_climate.pdf'
plt.tight_layout(pad=0.0, h_pad=0.5)
fig.set_size_inches(0.375 * productionplot.TEXTWIDTH, 3)
fig.savefig(plotname, bbox_inches='tight')
