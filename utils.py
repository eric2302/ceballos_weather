import requests
import pandas as pd

def flatten_dataframe(df, type='hourly'):
    
    # Get the number of rows in the dataframe (representing hours before the event)
    num_rows = df.shape[0]

    if type == 'hourly':
        idx = 'h'
    elif type == 'daily':
        idx = 'd'
    
    # Initialize a dictionary to store the new column names and values
    flattened_data = {}

    # Loop through each column (feature)
    for col in df.columns:
        # Loop through each row (hour before the event)
        for i in range(num_rows):
            # Create new column name by adding the hour suffix (e.g., 'feature_h-0', 'feature_h-1', etc.)
            new_col_name = f'{col}_{idx}-{num_rows - 1 - i}'
            # Add this column to the dictionary, with the value from the i-th row
            flattened_data[new_col_name] = [df.iloc[i][col]]

    # Convert the dictionary to a dataframe
    flattened_df = pd.DataFrame(flattened_data)

    return flattened_df

def get_lat_lon_from_web_service(easting, northing):
    """
    Get the latitude and longitude from the BGS web service for a given easting and northing co-ordinate.
    
    Parameters:
    easting (int): The easting co-ordinate.
    northing (int): The northing co-ordinate.
    
    Returns:
    latitude (float): The latitude co-ordinate.
    longitude (float): The longitude co-ordinate.
    """
    
    easting_full = easting * 100 + 50
    northing_full = northing * 100 + 50
    
    url = "http://webapps.bgs.ac.uk/data/webservices/CoordConvert_LL_BNG.cfc?method=BNGtoLatLng&" + \
         f"easting={easting_full}&northing={northing_full}"
         
    response = requests.get(url)
    if response.status_code == 200:
        # response is JSON dictionary, so return the key LATITUDE and LONGITUDE
        data = response.json()
        latitude = data["LATITUDE"]
        longitude = data["LONGITUDE"]
        return latitude, longitude
    else:
        raise Exception(f"Failed to retrieve data: HTTP Status Code {response.status_code}")