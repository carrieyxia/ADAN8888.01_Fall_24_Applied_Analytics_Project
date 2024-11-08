import pandas as pd
import os
from opencage.geocoder import OpenCageGeocode
from dotenv import load_dotenv


def get_longitude_latitude(location):
    key = os.getenv("OPENCAGE_API_KEY")
    geocoder = OpenCageGeocode(key)
    if location == 'Theatre District':
        lat = 42.3519
        long = -71.0643
        return lat, long

    else:
        query = location + ", Boston, MA"
        results = geocoder.geocode(query)
        if results:
            return results[0]['geometry']['lat'], results[0]['geometry']['lng']
        else:
            print(f"Location '{location}' not found.")
            return None, None


if __name__ == "__main__":
    # This code block will only run if the script is executed directly

    ################ Load data #################
    load_dotenv()

    current_dir = os.getcwd()

    data_folder = os.path.join(current_dir, "data")
    raw_data_folder = os.path.join(data_folder, "raw")
    interim_data_folder = os.path.join(data_folder, "interim")

    unique_combo_dir = os.path.join(interim_data_folder, 'unique_combo.csv')
    ride_locations_dir = os.path.join(interim_data_folder, "ride_locations.csv")

    locations = [
        'North End', 'West End', 'Beacon Hill', 'South Station',
        'North Station', 'Fenway', 'Boston University', 'Back Bay',
        'Theatre District', 'Northeastern University',
        'Financial District', 'Haymarket Square'
    ]
    
    location_dict = {}

    for location in locations:
        lat, long = get_longitude_latitude(location)
        if lat is not None and long is not None:
            location_dict[location] = [lat, long]  # Store in the dictionary
        else:
            print(f"Coordinates for '{location}' could not be found.")

    df = pd.read_csv(unique_combo_dir)

    # Apply the latitude and longitude values to each location in the dataframe
    df["source_lat"] = df['source'].apply(lambda x: location_dict[x][0] if x in location_dict else None)
    df["source_long"] = df['source'].apply(lambda x: location_dict[x][1] if x in location_dict else None)

    df["destination_lat"] = df['destination'].apply(lambda x: location_dict[x][0] if x in location_dict else None)
    df["destination_long"] = df['destination'].apply(lambda x: location_dict[x][1] if x in location_dict else None)

    # Save the updated dataframe to CSV
    df.to_csv(ride_locations_dir, index=False)
