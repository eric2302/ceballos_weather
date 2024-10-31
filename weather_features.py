# %%
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Daily
from utils import flatten_dataframe

# Access command-line arguments
i = sys.argv[1]

# use i to determine the batch
i = int(i)

# it's a multiple of 35875
start =  i * 35875
end = (i + 1) * 35875


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   Load assessment dates from UKBiobank
####################################################################################
ukb_sub = pd.read_csv('./data/ukb_dates.csv').iloc[start:end]
centre_coords = pd.read_csv('./data/ukb_centre_coords.csv')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           Generate weather data
################################################################################

# Initialize an empty list to store the results
results = []

# Iterate over each row in ukb_sub
# include progress bar
import tqdm
for i, row in tqdm.tqdm(ukb_sub.iterrows(), total=ukb_sub.shape[0]):
    
    centre_id = row['54-0.0']
    coord = centre_coords[centre_coords['centre_id'] == centre_id]
    lat = coord['lat'].values[0]
    lon = coord['lon'].values[0]

    # create Point object and fetch weather data
    point = Point(lat, lon)
    
    # check if there is a weather station
    if point.get_stations().empty:
        continue

    # extract date from ukb_sub. convert from string to year, month, day
    date = row['53-0.0'].split('-')
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    assessment_date = datetime(year, month, day)
    past_day = assessment_date - timedelta(days=5)
    
    
    # get daily data for the past days
    day_data = Daily(point, start=past_day, end=assessment_date).fetch()
    
    # keep only non-null data
    day_data = day_data.dropna(axis=1)

    # drop index
    day_data = day_data.reset_index(drop=True)

    # flatten data
    day_data = flatten_dataframe(day_data, type='daily')
    
    # concatenate to the main dataframe
    day_data = day_data.reset_index(drop=True)
    
    # # eid is the index of the df
    day_data.index = [row['eid']]
    
    results.append(day_data)

# loop through the results and concatenate them
results = pd.concat(results)

# rename index to 'eid'
results.index.name = 'eid'

# store the results
results.to_csv(f'./results/batch_data/ukb_weather_batch{i}.csv')

