# %%
import os, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyls import behavioral_pls
from scipy.stats import zscore, spearmanr

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       LOAD BODY DATA
###############################################################################
# load body features
body = pd.read_csv('./results/ukb_body_features.csv', index_col=0).iloc[:, :-2]

# load ukb variables
variable_names = pd.read_csv('data/ukb_variables.csv', index_col=1)

# rename columns to Description in ukb_variables
body.columns = body.columns.map(variable_names['Description'])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       LOAD WEATHER DATA
###############################################################################
# weather data comes in batches
# load and concatenate first half of batches
weather_files = sorted(glob.glob('./results/batch_data/ukb_weather_batch*.csv'))
nbatches = len(weather_files)
weather_files = weather_files[:nbatches//2]

# load and concatenate weather data
weather_discovery = pd.concat([pd.read_csv(f, index_col=0) for f in weather_files])

# match index of weather with body
body_discovery = body.loc[weather_discovery.index]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       PREPARE DATA FOR PLS
###############################################################################

# drop first measure of "Body mass index (BMI)"
# locate column index of "Body mass index (BMI)"
idx = np.nonzero(body_discovery.columns.get_loc('Body mass index (BMI)'))[0][0]
# drop column using boolean indexing
idx_mask = np.ones(body_discovery.shape[1], dtype=bool)
idx_mask[idx] = False
body_discovery = body_discovery.iloc[:, idx_mask]

# clean up data            
# filter columns with snow, prcp
weather_discovery = weather_discovery.drop(columns=weather_discovery.filter(like='prcp').columns)
weather_discovery = weather_discovery.drop(columns=weather_discovery.filter(like='snow').columns)
weather_discovery = weather_discovery.dropna()


# count number of missing observations in body_discovery
# and discard rows with more than 10% missing values
body_missing = body_discovery.isnull().sum(axis=1) / body_discovery.shape[1]

# retain rows of body_discovery with less than 10% missing values
body_discovery = body_discovery.loc[body_missing[body_missing < 0.1].index]

# discard columns with more than 5% missing values
body_missing = body_discovery.isnull().sum() / body_discovery.shape[0]
body_discovery = body_discovery.drop(columns=body_missing[body_missing > 0.05].index)

# fill body missing values with mean
body_discovery = body_discovery.fillna(body_discovery.mean())

# keep only rows that are in both body_discovery and weather_discovery
in_both = body_discovery.index.intersection(weather_discovery.index)
body_discovery = body_discovery.loc[in_both]
weather_discovery = weather_discovery.loc[in_both]

# # plot histograms of body_discovery columns
# fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
# body_discovery.hist(ax=ax, bins=50, alpha=0.7)
# fig.tight_layout()

# # plot histograms of weather_discovery columns
# fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
# weather_discovery.hist(ax=ax, bins=50, color='orange', alpha=0.7)
# fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                               PLS
###############################################################################

X = zscore(body_discovery, ddof=1)
Y = zscore(weather_discovery, ddof=1)
lv = 0

if os.path.exists('results/pls_result_discovery.npy'):
      pls_result = np.load('results/pls_result_discovery.npy', allow_pickle=True).item()
else:
      # do beahvioral PLS with body and weather discovery data
      pls_result = behavioral_pls(X, Y, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                              permindices=False, test_split=80, seed=0)
      np.save('results/pls_result_discovery.npy', pls_result)

# how many latent variables are significant?
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_perm = null_singvals**2 / sum(null_singvals**2)

# p-value
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+1000)

# plot cv
fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 3))
sns.boxplot(cv_perm.T * 100, color='lightgreen', fliersize=0, zorder=1)
sns.scatterplot(x=np.arange(len(cv)), y=cv * 100, color='orange', size=10, ax=ax, zorder=2, legend=False)
ax.set_ylabel('Covariance explained (%)')
ax.set_xlabel('Latent variable')
sns.despine()
ax.set_xticklabels([str(i) for i in np.arange(len(cv))+1], rotation=90)
fig.tight_layout()

# plot scores
fig, ax = plt.subplots(1, 1, dpi=200)
sns.regplot(x=pls_result['x_scores'][:,lv], y=pls_result['y_scores'][:,lv], ax=ax,
                scatter_kws={'s': 1}, line_kws={'color': 'black', 'linewidth': 1})
ax.set_xlabel('Body health scores')
ax.set_ylabel('Weather scores')
sns.despine()

# plot weather loadings
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2

plot_df = pd.DataFrame({'loading': pls_result['y_loadings'][:, lv],
                        'err': err,
                        'feature': weather_discovery.columns})
plot_df['sign'] = np.sign(plot_df['loading'])
plot_df = plot_df.sort_values('loading', ascending=False)

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10,5))
sns.barplot(x='feature', y='loading', data=plot_df, ax=ax, errorbar=None, 
            hue='sign', palette='tab10')
ax.errorbar(plot_df['feature'], plot_df['loading'], yerr=plot_df['err'], linestyle='None', color='grey')
ax.get_legend().remove()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()

# do PLS with X and Y switched to get loadings for body
if os.path.exists('results/pls_X_result_discovery.npy'):
      pls_result_X = np.load('results/pls_X_result_discovery.npy', allow_pickle=True).item()
else:
      # do beahvioral PLS with body and weather discovery data
      pls_result_X = behavioral_pls(Y, X, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                              permindices=False, test_split=80, seed=0)
      np.save('results/pls_X_result_discovery.npy', pls_result_X)

# # switch X and Y to get loadings for body
# pls_result_X = behavioral_pls(Y, X, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
#                               permindices=False, test_split=80, seed=0)

# plot body loadings
err = (pls_result_X["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result_X["bootres"]["y_loadings_ci"][:, lv, 0]) / 2

plot_df = pd.DataFrame({'loading': pls_result_X['y_loadings'][:, lv],
                        'err': err,
                        'feature': body_discovery.columns})
plot_df['sign'] = np.sign(plot_df['loading'])
plot_df = plot_df.sort_values('loading', ascending=False)

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10,5))
sns.barplot(x='feature', y='loading', data=plot_df, ax=ax, errorbar=None,
            hue='sign', palette='tab10')
ax.errorbar(plot_df['feature'], plot_df['loading'], yerr=plot_df['err'], linestyle='None', color='grey')
ax.get_legend().remove()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        LOAD REPLICATION DATA
###############################################################################
# replication is other half of batches
weather_files = sorted(glob.glob('./results/batch_data/ukb_weather_batch*.csv'))
weather_files = weather_files[nbatches//2:]

# load and concatenate weather data
weather_replication = pd.concat([pd.read_csv(f, index_col=0) for f in weather_files])

# match index of weather with body
body_replication = body.loc[weather_replication.index]

# discard BMI column
idx = np.nonzero(body_replication.columns.get_loc('Body mass index (BMI)'))[0][0]
idx_mask = np.ones(body_replication.shape[1], dtype=bool)
idx_mask[idx] = False
body_replication = body_replication.iloc[:, idx_mask]

# clean up data
weather_replication = weather_replication.drop(columns=weather_replication.filter(like='prcp').columns)
weather_replication = weather_replication.drop(columns=weather_replication.filter(like='snow').columns)
weather_replication = weather_replication.dropna()
body_missing = body_replication.isnull().sum(axis=1) / body_replication.shape[1]
body_replication = body_replication.loc[body_missing[body_missing < 0.1].index]
body_missing = body_replication.isnull().sum() / body_replication.shape[0]
body_replication = body_replication.drop(columns=body_missing[body_missing > 0.05].index)
body_replication = body_replication.fillna(body_replication.mean())
in_both = body_replication.index.intersection(weather_replication.index)
body_replication = body_replication.loc[in_both]
weather_replication = weather_replication.loc[in_both]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        REPLICATION PLS
###############################################################################
X = zscore(body_replication, ddof=1)
Y = zscore(weather_replication, ddof=1)
lv = 0

if os.path.exists('results/pls_result_replication.npy'):
      pls_result_rep = np.load('results/pls_result_replication.npy', allow_pickle=True).item()
else:
      # do beahvioral PLS with body and weather discovery data
      pls_result_rep = behavioral_pls(X, Y, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                                    permindices=False, test_split=80, seed=0)
      np.save('results/pls_result_replication.npy', pls_result_rep)

# compare weights across discovery and replication
x_weights_discovery = pls_result['x_weights'][:, lv]
x_weights_replication = pls_result_rep['x_weights'][:, lv]
y_weights_discovery = pls_result['y_weights'][:, lv]
y_weights_replication = pls_result_rep['y_weights'][:, lv]

# correlation between discovery and replication weights
r_x = np.corrcoef(x_weights_discovery, x_weights_replication)[0, 1]
r_y = np.corrcoef(y_weights_discovery, y_weights_replication)[0, 1]
print(f'Correlation between discovery and replication weights for body health features: {r_x:.4f}')
print(f'Correlation between discovery and replication weights for weather features: {r_y:.4f}')

# plot regplot of replication and discovery weights
fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 5))
sns.regplot(x=x_weights_discovery, y=x_weights_replication, ax=ax[0], scatter_kws={'s': 1},
            line_kws={'color': 'black', 'linewidth': 1, 'alpha': 0.5}, ci=None)
ax[0].set_xlabel('Discovery loadings')
ax[0].set_ylabel('Replication loadings')
ax[0].set_title('Body health features')
sns.regplot(x=y_weights_discovery, y=y_weights_replication, ax=ax[1], scatter_kws={'s': 1},
            line_kws={'color': 'black', 'linewidth': 1, 'alpha': 0.5}, ci=None)
ax[1].set_xlabel('Discovery loadings')
ax[1].set_ylabel('Replication loadings')
ax[1].set_title('Weather features')
fig.tight_layout()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     OUT-OF-SAMPLE SCORE CORRELATION
###############################################################################
# get comparable scores using discovery weights
scores_y = Y @ pls_result["y_weights"][:, lv]
scores_x = X @ pls_result["x_weights"][:, lv]

# # make sure both are equally sized
# n_obs = min(len(scores_dis), len(scores_rep))
# scores_dis, scores_rep = scores_dis[:n_obs], scores_rep[:n_obs]

oos_corr, p = spearmanr(scores_x, scores_y)
is_corr, p = spearmanr(pls_result["x_scores"][:, lv], pls_result["y_scores"][:, lv])

# regplot of out-of-sample scores
fig, ax = plt.subplots(1, 1, dpi=200)
sns.regplot(x=scores_x, y=scores_y, ax=ax, scatter_kws={'s': 1},
            line_kws={'linewidth': 5, 'alpha': 0.5}, ci=None)

# plot scores of discovery
sns.regplot(x=pls_result["x_scores"][:,lv], y=pls_result["y_scores"][:,lv], ax=ax, scatter=True, 
            line_kws={'color': 'orange', 'linewidth': 5, 'alpha': 0.5}, ci=None, scatter_kws={'s': 1, 'color': 'orange'})
ax.set_xlabel('X scores')
ax.set_ylabel('Y scores')
ax.set_title(f'In-sample correlation: {is_corr:.2f} | Out-of-sample correlation: {oos_corr:.2f} ')

# add legend with blue and orange colorbars
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)]
labels = ['Replication', 'Discovery']
ax.legend(handles, labels, frameon=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        CORRELATE TAVG TO VITAMIN D
###############################################################################

# load tavg from weather discovery
tavg = weather_discovery['tavg_d-0']
vitamin_d = body_discovery['Vitamin D']

# filter out rows with missing values
tavg = tavg.loc[~tavg.isnull()]
vitamin_d = vitamin_d.loc[~vitamin_d.isnull()]

# keep rows where both tavg and vitamin_d are not missing
rows_to_keep = tavg.index.intersection(vitamin_d.index)
tavg = tavg.loc[rows_to_keep]
vitamin_d = vitamin_d.loc[rows_to_keep]

# compute spearmans correlation
r, p = spearmanr(tavg, vitamin_d)

# scatter plot
fig, ax = plt.subplots(1, 1, dpi=200)
sns.regplot(x=tavg, y=vitamin_d, ax=ax, scatter_kws={'s': .1, 'color': 'grey', 'alpha': 0.5},
            line_kws={'color': 'black', 'linewidth': 1})
ax.set_xlabel('Average daily temperature (°C)')
ax.set_ylabel('Vitamin D (nmol/L)')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        TURN VARIABLES INTO LATEX TABLE
###############################################################################
variable_names = pd.read_csv('data/ukb_variables.csv')

# convert to latex table
variable_names.to_latex('results/ukb_variables.tex', index=False)
variable_names = pd.read_csv('data/ukb_variables.csv', index_col=1)
