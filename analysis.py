# %%
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyls import behavioral_pls
from scipy.stats import zscore

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

                                     
# filter columns with snow, prcp
weather_discovery = weather_discovery.drop(columns=weather_discovery.filter(like='prcp').columns)
weather_discovery = weather_discovery.drop(columns=weather_discovery.filter(like='snow').columns)

# count number of missing values in both sets
body_missing = body_discovery.isnull().sum() / body_discovery.shape[0]
weather_missing = weather_discovery.isnull().sum() / weather_discovery.shape[0]

# discard columns with more than 10% missing values
body_discovery = body_discovery.drop(columns=body_missing[body_missing > 0.1].index)

# keep weather rows that contain at least 90% non-NaN values
weather_missing = weather_discovery.isnull().sum(axis=1) / weather_discovery.shape[1]

# index rows of weather_discovery with less than 10% missing values
weather_discovery = weather_discovery.loc[weather_missing[weather_missing < 0.1].index]

# fill weather missing values with mean
weather_discovery = weather_discovery.fillna(weather_discovery.mean())

# keep body rows that are in weather_discovery
body_discovery = body_discovery.loc[weather_discovery.index]

# fill body missing values with mean
body_discovery = body_discovery.fillna(body_discovery.mean())


# # plot histograms of body_discovery columns
# fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
# body_discovery.hist(ax=ax, bins=50, alpha=0.7)
# fig.tight_layout()

# # plot histograms of weather_discovery columns with tightened layout
# fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
# weather_discovery.hist(ax=ax, bins=50, color='orange', alpha=0.7)
# fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                               PLS
###############################################################################

X = zscore(body_discovery, ddof=1)
Y = zscore(weather_discovery, ddof=1)
lv = 0

# do beahvioral PLS with body and weather discovery data
pls_result = behavioral_pls(X, Y, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                            permindices=False, test_split=80, seed=0)


# plot scores
fig, ax = plt.subplots(1, 1, dpi=200)
sns.regplot(x=pls_result['x_scores'][:,lv], y=pls_result['y_scores'][:,lv], ax=ax,
                scatter_kws={'s': 1})

# plot loadings
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2

plot_df = pd.DataFrame({'loading': pls_result['y_loadings'][:, lv],
                        'err': err,
                        'feature': weather_discovery.columns})
plot_df['sign'] = np.sign(plot_df['loading'])
plot_df = plot_df.sort_values('loading', ascending=False)

fig, ax = plt.subplots(1, 1, dpi=200)
sns.barplot(x='feature', y='loading', data=plot_df, ax=ax, errorbar=None, 
            hue='sign', palette='tab10')
ax.errorbar(plot_df['feature'], plot_df['loading'], yerr=plot_df['err'], linestyle='None', color='grey')

# format plot
ax.get_legend().remove()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        REPLICATION SET
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
body_missing = body_replication.isnull().sum() / body_replication.shape[0]
weather_missing = weather_replication.isnull().sum() / weather_replication.shape[0]
body_replication = body_replication.drop(columns=body_missing[body_missing > 0.1].index)
weather_missing = weather_replication.isnull().sum(axis=1) / weather_replication.shape[1]
weather_replication = weather_replication.loc[weather_missing[weather_missing < 0.1].index]
weather_replication = weather_replication.fillna(weather_replication.mean())
body_replication = body_replication.loc[weather_replication.index]
body_replication = body_replication.fillna(body_replication.mean())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        REPLICATION PLS
###############################################################################
X = zscore(body_replication, ddof=1)
Y = zscore(weather_replication, ddof=1)
lv = 0

# do beahvioral PLS with body and weather discovery data
pls_result_rep = behavioral_pls(X, Y, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                                permindices=False, test_split=80, seed=0)

# %%
# compare weights across discovery and replication
x_weights_discovery = pls_result['x_weights'][:, lv]
x_weights_replication = pls_result_rep['x_weights'][:, lv]
y_weights_discovery = pls_result['y_weights'][:, lv]
y_weights_replication = pls_result_rep['y_weights'][:, lv]

# plot regplot of replication and discovery weights
fig, ax = plt.subplots(1, 1, dpi=200)
sns.regplot(x=x_weights_discovery, y=x_weights_replication, ax=ax, scatter_kws={'s': 1})
sns.regplot(x=y_weights_discovery, y=y_weights_replication, ax=ax, scatter_kws={'s': 1})

