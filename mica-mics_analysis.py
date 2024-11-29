# %%
import glob
from random import seed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from pyls import behavioral_pls

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       LOAD BRAIN DATA
###############################################################################
# load data/parcellations/lut_schaefer-100_mics.csv
subcortex_lut = pd.read_csv('./data/parcellations/lut_subcortical-cerebellum_mics.csv').iloc[:14]
cortex_lut = pd.read_csv('./data/parcellations/lut_schaefer-100_mics.csv')
# merge
lut = pd.concat([subcortex_lut, cortex_lut], axis=0).reset_index(drop=True)

# find index of row where label=='medial_wall'
to_discard = lut[lut['label']=='medial_wall'].index.values

# load conectomes for all subjects in data
files = glob.glob('./data/mica-mics/sub-*_ses-*_space-fsnative_atlas-schaefer100_desc-fc.txt')

fcs = []

for file in files:
    data = np.loadtxt(file, delimiter=',')
    
    # discard medial wall rows and columns
    data = np.delete(data, to_discard, axis=0)
    data = np.delete(data, to_discard, axis=1)   
    
    # mirror the matrix around the diagonal
    np.fill_diagonal(data, 0)
    data = data + data.T
    
    # sum strength of functional connectivity 
    data = data.sum(axis=0)
    fcs.append(data)

fcs = np.array(fcs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       LOAD WEATHER DATA AND RUN PLS
###############################################################################
weather = pd.read_csv('./data/mica-mics/mica_weather_features.csv')

X = zscore(fcs, ddof=1)
Y = zscore(weather.values, ddof=1)
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
nperm = 1000
pls_result = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=None,
                            permindices=False, test_split=0, seed=0)

pls_result_X = behavioral_pls(Y, X, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=None,
                              permindices=False, test_split=0, seed=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       LOAD WEATHER DATA AND RUN PLS
###############################################################################
lv = 0
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_perm = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

# covariance explained
fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 3))
sns.boxplot(cv_perm.T * 100, color='lightgreen', fliersize=0, zorder=1)
sns.scatterplot(x=np.arange(len(cv)), y=cv * 100, color='orange', size=10, ax=ax, zorder=2, legend=False)
ax.set_ylabel('Covariance explained (%)')
ax.set_xlabel('Latent variable')
sns.despine()
ax.set_xticklabels([str(i) for i in np.arange(len(cv))+1], rotation=90)
fig.tight_layout()

# score plot
xscore = pls_result["x_scores"][:, lv]
yscore = pls_result["y_scores"][:, lv]
scores = pd.DataFrame({'FC scores': xscore, 
                        'Weather scores': yscore})

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 6))
sns.regplot(data=scores, x='FC scores', y='Weather scores', ax=ax)
sns.despine()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       BRAIN LOADINGS
###############################################################################
# generate list of region names
roi_names = lut['label'].str.split('_').apply(lambda x: ' '.join(x[1:]))
roi_names[:14] = subcortex_lut['label'].values
roi_names = roi_names[roi_names != 'wall']

networks = lut['label'].str.split('_').str.get(2)
networks[:14] = ['Subcortex'] * 14
networks = networks[~networks.isna()]

# create dataframe with loadings and errors
err = (pls_result_X["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result_X["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
fc_df = pd.DataFrame({'region': roi_names,
                      'network': networks,
                      'loading': pls_result_X["y_loadings"][:, lv],
                      'err': err})
fc_df['sign'] = np.sign(fc_df['loading'])
fc_df = fc_df.sort_values('loading', ascending=False)

tab10_orange = sns.color_palette('tab10')[1]

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(18,8))
sns.barplot(x='region', y='loading', data=fc_df, ax=ax, errorbar=None, palette='tab10', hue='network')
ax.errorbar(fc_df['region'], fc_df['loading'], yerr=fc_df['err'], linestyle='None', color='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()
# move legend to the right
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()

# average loadings and err across regions for each network
fc_net_df = fc_df.groupby(['network']).agg({'loading': 'mean', 'err': 'mean'}).reset_index()
fc_net_df = fc_net_df.sort_values('loading', ascending=False)

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6,3))
sns.barplot(x='network', y='loading', data=fc_net_df, ax=ax, errorbar=None, palette='tab10', hue='network')
ax.errorbar(fc_net_df['network'], fc_net_df['loading'], yerr=fc_net_df['err'], linestyle='None', color='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       WEATHER LOADINGS
###############################################################################
# create dataframe with loadings and errors
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
weather_df = pd.DataFrame({'feature': weather.columns, 
                      'loading': pls_result["y_loadings"][:, lv],
                        'err': err})
weather_df['sign'] = np.sign(weather_df['loading'])  
weather_df = weather_df.sort_values('loading', ascending=False)

# remove features that contain 'rhum' or 'dwpt'
weather_df = weather_df[~weather_df['feature'].str.contains('rhum|dwpt')]

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(7,5))
sns.barplot(x='feature', y='loading', data=weather_df, ax=ax, errorbar=None, 
            hue='sign', palette='tab10')
ax.errorbar(weather_df['feature'], weather_df['loading'], yerr=weather_df['err'], linestyle='None', color='grey')
ax.get_legend().remove()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()
fig.tight_layout()
