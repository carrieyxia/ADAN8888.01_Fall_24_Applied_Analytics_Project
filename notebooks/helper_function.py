import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, root_mean_squared_error


################## Data Partition ##################
def train_val_test_split(df):
    """
    Splits a DataFrame into training, validation, and test sets.

    The dataset is shuffled before splitting. The validation set is 20% 
    and the test set is 10% of the total data. The remaining data is used 
    as the training set.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.

    Returns:
        tuple: A tuple containing three DataFrames:
            - train_df (pd.DataFrame): The training set.
            - val_df (pd.DataFrame): The validation set.
            - test_df (pd.DataFrame): The test set.
    """
    # Shuffle the dataset and calculate the size of validation and test sets

    df = df.sample(frac=1, random_state=123)

    val_size = int(len(df) * 0.2)
    test_size = int(len(df) * 0.1)

    # Select rows based on the val_size and test_size to store as train set, val set, and test set
    train_df = df.iloc[val_size + test_size:]
    val_df = df.iloc[:val_size]
    test_df = df.iloc[val_size:val_size + test_size]
    return train_df, val_df, test_df

################## Feature Engineering ##################
def weather_severity(row):
    """
    Determines the weather severity based on weather icon, summaries, and visibility.

    Args:
        row (pd.Series): A row of weather data containing 'icon', 'short_summary', 
                         'long_summary', and 'visibility'.

    Returns:
        int: The severity level (1 = low, 2 = moderate, 3 = high).
    """
    # Initialize severity
    severity = 1

    if 'rain' in row['icon'].strip().lower():
        # If rain is present in the icon, check for light or drizzle
        if 'light' in row['short_summary'].lower() or 'drizzle' in row['short_summary'].lower() or \
           'light' in row['long_summary'].lower() or 'drizzle' in row['long_summary'].lower():
            severity = 2  # Moderate severity for light rain or drizzle
        else:
            severity = 3  # Highest severity for rain without light or drizzle
    elif 'cloudy' in row['icon'].lower() or 'fog' in row['icon'].lower():
        severity = 2  # Moderate severity for clouds and fog

    # Adjust severity based on visibility
    if row['visibility'] < 1:  # Low visibility (less than 1)
        severity += 1  # Increase severity by 1
    elif row['visibility'] >= 7:  # High visibility (7 or more)
        severity -= 1  # Decrease severity by 1
        severity = max(severity, 1)  # Ensure severity doesn't go below 1

    return severity

def add_time_features(df):
    """
    Adds time-related features to the DataFrame.

    Features added:
        - rush_hour: 1 if the hour is during morning or evening rush hours (6-9 AM or 4-6 PM), otherwise 0.
        - weekend: 1 if the day is Saturday or Sunday, otherwise 0.
        - game_day: 1 if the date corresponds to a Bruins or Celtics game day, otherwise 0.

    Args:
        df (pd.DataFrame): DataFrame with 'datetime' column (as datetime) and 'hour' column.

    Returns:
        pd.DataFrame: The input DataFrame with added features.
    """
    # Create rush_hour feature
    df['rush_hour'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9 or 16 <= x <= 18) else 0)

    # Create weekend feature
    df['weekend'] = df['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Define game dates for Bruins and Celtics
    bruins = [
        '2018-11-05', '2018-11-08', '2018-11-10', '2018-11-11',
        '2018-11-23', '2018-11-29', '2018-12-01', '2018-12-08',
        '2018-12-11', '2018-12-16', '2018-12-20', '2018-12-22',
        '2018-12-27'
    ] 
    celtics = [
        '2018-11-01', '2018-11-14', '2018-11-16', '2018-11-17',
        '2018-11-21', '2018-11-30', '2018-12-06', '2018-12-10',
        '2018-12-14', '2018-12-19', '2018-12-21', '2018-12-23',
        '2018-12-25'
    ]
    game_dates = bruins + celtics

    # Create game_day feature
    df['game_day'] = df['datetime'].apply(lambda x: 1 if x.strftime('%Y-%m-%d') in game_dates else 0)

    return df

################## Data Preprocessing ##################
## Missing Value Imputation
def shift_dt(df):
    """
    Converts the 'datetime' column to the format required by the Mapbox API.

    This function adjusts the 'datetime' column by:
    1. Rounding down to the nearest hour.
    2. Formatting it as a string in the ISO-like format: `YYYY-MM-DDTHH:00`.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'datetime' column.

    Returns:
        pd.DataFrame: The modified DataFrame with 'datetime' formatted for Mapbox API requests.
    """
    df['datetime'] = pd.to_datetime(df['datetime']).dt.floor('h')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:00')

    return df

def taxi_price_calculator(distance, time):
    base_fare = 2.60
    per_min_fare = 0.47
    per_mile_fare = 2.8
    price = base_fare + distance * per_mile_fare + time * per_min_fare
    return price

def calculate_taxi_price(df: pd.DataFrame, time_df: pd.DataFrame, rush_hour = False):
    """
    Calculates and updates taxi prices in a DataFrame based on distance and estimated time of arrival (ETA).

    This function merges the input DataFrame (`df`) with a DataFrame that contains eta_minutes pulled from Mapbox API (`time_df`) and calculates the taxi prices based on distance and ETA. It handles different pricing strategies based on whether it is rush hour.

    Args:
        df (pd.DataFrame): 
            The main DataFrame containing information about trips, including the source, destination, and trip details.
        time_df (pd.DataFrame): 
            A supplementary DataFrame with time-specific details, such as ETA and other time-based information, 
            to be merged with the main DataFrame.
        rush_hour (bool, optional): 
            A flag indicating whether the pricing should consider rush hour. Defaults to `False`.

    Returns:
        pd.DataFrame: 
            The updated DataFrame with taxi prices calculated and added under the 'price' column for rows 
            where `name` is 'Taxi'.

    Notes:
        - The `taxi_price_calculator` function is assumed to compute the taxi price based on the `distance` 
          and `eta_minutes` columns.
        - During rush hour, the `datetime` column is also considered in the merge operation.
        - This function assumes that `df` contains a column named `name` to filter rows for taxi-related entries.
    """
    if rush_hour:
         df = pd.merge(df, time_df, on=['source', 'destination', 'datetime'], how = 'left')
         df.loc[df['name'] == 'Taxi', 'price'] = df.loc[df['name']== 'Taxi'].apply(lambda row: taxi_price_calculator(row['distance'], row['eta_minutes']), axis = 1)
    else:
        df = pd.merge(df, time_df, on=['source', 'destination'], how = 'left')
        df.loc[df['name'] == 'Taxi', 'price'] = df.loc[df['name']== 'Taxi'].apply(lambda row: taxi_price_calculator(row['distance'], row['eta_minutes']), axis = 1)

    return df

## Encoding Categorical Variables
class OneHotEncodingProcessor:
    def __init__(self, column_name, categories = None):
        """
        Initialize the processor with the column to be one-hot encoded.
        """
        self.column_name = column_name
        self.categories = [categories] if categories is not None else 'auto'
        self.encoder = OneHotEncoder(categories=self.categories, sparse_output=False, handle_unknown='ignore')

    def preprocess(self, df):
        """
        Preprocess the input DataFrame by stripping and replacing spaces with underscores in the target column.
        """
        df[self.column_name] = df[self.column_name].str.strip().str.replace(' ', '_')
        return df

    def fit(self, train_df):
        """
        Fit the encoder on the training data.
        """
        self.preprocess(train_df)
        self.encoder.fit(train_df[[self.column_name]])

    def transform(self, df):
        """
        Transform the input DataFrame using the trained encoder and concatenate the one-hot encoded columns.
        """
        self.preprocess(df)
        encoded_data = self.encoder.transform(df[[self.column_name]])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out([self.column_name]))
        return pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

################## Model Evaluation ##################

def evaluate_model(model, X, y_true):
    """
    Evaluate a regression model on a given dataset.
    
    Parameters:
    - model: Trained regression model
    - X: Features for predictions
    - y_true: True target values
    
    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - r2: R-squared score
    """
    # Predict using the model
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, r2
def evaluate_model_v2(y_true, y_pred, individual_cases = False):
    """
    Evaluate a regression model's performance.

    Parameters:
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    individual_cases : bool, optional, default=False
        If True, returns MSE and RMSE. If False, includes R² score.

    Returns:
    -------
    tuple
        - If `individual_cases` is True: (mse, rmse)
        - If `individual_cases` is False: (mse, rmse, r2)
    """
    if individual_cases == True:
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = rmse**2
        return mse, rmse

    else:
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = rmse**2
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, r2

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = rmse**2
    return mse

################## Model Deployment ##################
# Save cleaned data to CSV
def save_data(df, file_path):
    """
    Save a DataFrame to a CSV file.
    Dependencies: pandas
    """
    df.to_parquet(file_path, index=False)
    print(f"Data saved to {file_path}")

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    Dependencies: pandas
    """
    df = pd.read_parquet(file_path)
    print(f"Data loaded from {file_path}")
    return df

def save_model(model, file_path):
    """
    Serialize and save the model to a file.
    Dependencies: pickle
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load the serialized model from a file.
    Dependencies: pickle
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_path}")
    return model