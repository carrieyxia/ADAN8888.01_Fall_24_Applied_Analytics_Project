
from geopy.geocoders import Nominatim
import pandas as pd
import os

def get_longtitude_latitude(location):
    # calling the Nominatim tool and create Nominatim class
    loc = Nominatim(user_agent="Geopy Library")
    location = location+", Boston"
    getLoc = loc.geocode(location)
    return getLoc.latitude, getLoc.longitude




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

    locations = ['North End', 'West End', 'Beacon Hill', 'South Station',
       'North Station', 'Fenway', 'Boston University', 'Back Bay',
       'Theatre District', 'Northeastern University',
       'Financial District', 'Haymarket Square']
    

    location_dict = {}

    for location in locations:
        lat, long = get_longtitude_latitude(location)
        if lat is not None and long is not None:
            location_dict[location] = [lat, long]  # Store in the dictionary
        # else:
        #     print(f"Location '{location}' not found.")

    df = pd.read_csv(unique_combo_dir)

    df["source_lat"] = df['source'].apply(lambda x: location_dict[x][0])
    df["source_long"] = df['source'].apply(lambda x: location_dict[x][1])

    df["destination_lat"] = df['destination'].apply(lambda x: location_dict[x][0])
    df["destination_long"] = df['destination'].apply(lambda x: location_dict[x][1])


    df.to_csv(ride_locations_dir, index = False)




