# %%
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyls import behavioral_pls
from scipy.stats import zscore

# load body features
body = pd.read_csv('./results/ukb_body_features.csv', index_col=0).iloc[:, :-2]

# load ukb variables
variable_names = pd.read_csv('data/ukb_variables.csv', index_col=1)

# rename columns to Description in ukb_variables
body.columns = body.columns.map(variable_names['Description'])

# %%
# load first batch of weather data
weather = pd.read_csv('./results/batch_data/ukb_weather_batch0.csv', index_col=0)

# discard rows where all NaNs
weather = weather.dropna(how='all')

# drop columns with all NaNs
weather = weather.dropna(axis=1, how='all')

# drop prcp wdir and snow columns. use wildcard filter
weather = weather.drop(columns=weather.filter(like='prcp').columns)
weather = weather.drop(columns=weather.filter(like='wdir').columns)
weather = weather.drop(columns=weather.filter(like='snow').columns)

# keep rows that have pres 
weather = weather.dropna(subset=weather.filter(like='pres').columns)

# %%
# correlate weather data with corresponding body features
# match index of weather with body
body_subset = body.loc[weather.index]

# count number of missing values in body_subset
missing = body_subset.isnull().sum() / body_subset.shape[0]

# %%
# from body_subset, drop columns 100021-0.0 and 100024-0.0
body_subset = body_subset.drop(columns=['100021-0.0', '100024-0.0'])
# %%
pulse = pd.read_csv('/poolz2/nnl-data/project-release-data/UKBiobank/Tabular/676574/ukb676574.csv', usecols=[442])
# add pulse to body
body = pd.concat([body, pulse], axis=1)
# redorder columns alphabetically
body = body.sort_index(axis=1)

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

# plot histograms of body_discovery columns
fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
body_discovery.hist(ax=ax, bins=50, alpha=0.7)
fig.tight_layout()

# plot histograms of weather_discovery columns with tightened layout
fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=200)
weather_discovery.hist(ax=ax, bins=50, color='orange', alpha=0.7)
fig.tight_layout()

# keep weather rows that contain at least 90% non-NaN values
weather_missing = weather_discovery.isnull().sum(axis=1) / weather_discovery.shape[1]

# index rows of weather_discovery with less than 10% missing values
weather_discovery = weather_discovery.loc[weather_missing[weather_missing < 0.1].index]

# fill weather missing values with median
weather_discovery = weather_discovery.fillna(weather_discovery.mean())

# keep body rows that are in weather_discovery
body_discovery = body_discovery.loc[weather_discovery.index]

# fill body missing values with median
body_discovery = body_discovery.fillna(body_discovery.mean())


# %%
X = zscore(body_discovery, ddof=1)
Y = zscore(weather_discovery, ddof=1)

# do beahvioral PLS with body and weather discovery data
pls_result = behavioral_pls(X, Y, n_boot=1000, n_perm=1000, rotate=True, permsamples=None,
                            permindices=False, test_split=80, seed=0)


