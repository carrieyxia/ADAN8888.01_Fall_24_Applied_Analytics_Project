from geopy.geocoders import Nominatim
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Access the API key
mapbox_api_key = os.getenv('MAPBOX_API_KEY')
print(mapbox_api_key)

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
    current_dir = os.getcwd()

    data_folder = os.path.join(current_dir ,"data")
    raw_data_folder = os.path.join(data_folder,"raw")
    interim_data_folder = os.path.join(data_folder,"interim")

    unique_combo_dir = os.path.join(interim_data_folder, 'unique_combo.csv')
    ride_locations_dir = os.path.join(interim_data_folder, "ride_locations.csv")
    rides_with_eta_dir = os.path.join(interim_data_folder, "rides_with_etas.csv")

    df = pd.read_csv(ride_locations_dir)
    df['eta_minutes'] = df.apply(lambda row: get_eta(row['source_lat'], row['source_long'], 
                                                 row['destination_lat'], row['destination_long']), axis=1)
    

    df[['source', 'destination','eta_minutes']].to_csv(rides_with_eta_dir, index = False)