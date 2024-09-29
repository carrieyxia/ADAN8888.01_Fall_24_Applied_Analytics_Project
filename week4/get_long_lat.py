
from geopy.geocoders import Nominatim
import pandas as pd

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

    df = pd.read_csv('/Users/carriexia/Documents/GitHub/ADAN8888.01_Fall_24_Applied_Analytics_Project/week4/unique_combo.csv')

    df["source_lat"] = df['source'].apply(lambda x: location_dict[x][0])
    df["source_long"] = df['source'].apply(lambda x: location_dict[x][1])

    df["destination_lat"] = df['destination'].apply(lambda x: location_dict[x][0])
    df["destination_long"] = df['destination'].apply(lambda x: location_dict[x][1])

    df.to_csv("/Users/carriexia/Documents/GitHub/ADAN8888.01_Fall_24_Applied_Analytics_Project/week4/ride_locations.csv")


# Print the resulting dictionary


