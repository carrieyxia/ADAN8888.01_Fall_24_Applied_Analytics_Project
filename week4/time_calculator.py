from geopy.geocoders import Nominatim
import pandas as pd

# 1. get the longtitude and latitude for the locations
# 2. save the unique combo as a csv
# 3. merge two table: create sourece latitude, soure longtitude, desitination latitude, destination longtitude
# 4. use the Map API to pull time estimated by driving
# 5. save the csv with source, destination, time_estimated
# importing geopy library and Nominatim class

import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Access the API key
mapbox_api_key = os.getenv('MAPBOX_API_KEY')

# Now you can use the API key in your requests
print(mapbox_api_key)  # Just to verify it's loaded c

def get_eta(source_lat, source_long, destination_lat, destination_long, mode='driving'):

    
    mapbox_token = mapbox_api_key
    
    url = f"https://api.mapbox.com/directions/v5/mapbox/{mode}/{source_long},{source_lat};{destination_long},{destination_lat}"
    params = {
        'access_token': mapbox_token,
        'geometries': 'geojson', 
        'overview': 'full',
        'steps': 'true',
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract travel time in seconds (can be converted to minutes)
    travel_time_seconds = data['routes'][0]['duration']
    
    # Convert seconds to minutes
    travel_time_minutes = travel_time_seconds / 60
    
    return travel_time_minutes





if __name__ == "__main__":
    # This code block will only run if the script is executed directly
    # It won't run if the script is imported as a module in another script

    ################ load data #################
    df = pd.read_csv("week4/ride_locations.csv")
    df['eta_minutes'] = df.apply(lambda row: get_eta(row['source_lat'], row['source_long'], 
                                                 row['destination_lat'], row['destination_long']), axis=1)
    


    df[['source', 'destination','eta_minutes']].to_csv("/Users/carriexia/Documents/GitHub/ADAN8888.01_Fall_24_Applied_Analytics_Project/data/saved/rides_with_etas.csv")