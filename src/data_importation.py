import glob

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from src.assert_statements import assert_df
from tqdm import tqdm


def _load_and_process_csv(file):
    """
    Load and process the given CSV file.

    Params
    ------
    - file (str): The CSV file to load.

    Returns
    -------
    - pd.DataFrame: The loaded and processed data. If the file is not found, an empty DataFrame is returned.
    """
    df = pd.read_csv(file)
    if df is not None:
        return df
    else:
        return pd.DataFrame()


def load_data(
    dir: str = "../../downloaded_files",
    column_sort: list = ["Admin2", "Start_Date", "End_Date", "Flooded_Area_SqKM"],
    column_asc: list = [True, True, True, False],
) -> pd.DataFrame:
    """
    Load data from the given directory and sort it based on the given columns.

    Params
    ------
    - dir (str): The directory where the data is stored. Default is "../downloaded_files".
    - column_sort (list): The columns to sort the data by. Default is ["Admin2", "Start_Date", "End_Date", "Flooded_Area_SqKM"].
    - column_asc (list): The order of the columns to sort the data by. Default is [True, True, True, False].

    Returns
    -------
    - pd.DataFrame: The sorted data.
    """
    assert isinstance(dir, str), "Directory is not a string."
    assert isinstance(column_sort, list), "column_sort is not a list."
    assert isinstance(column_asc, list), "column_asc is not a list."

    files = glob.glob(f"{dir}/*.csv")
    assert len(files) > 0, "No files found in the given directory."

    data_frames = Parallel(n_jobs=-1)(
        delayed(_load_and_process_csv)(file)
        for file in tqdm(files, desc="Loading dataframes", unit="file")
    )

    data = pd.concat(data_frames)
    data.sort_values(by=column_sort, ascending=column_asc, inplace=True)
    data.reset_index(drop=True, inplace=True)

    assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."
    assert data.shape[0] > 0, "No data found in the given directory."
    assert data.shape[1] > 0, "No columns found in the given directory."
    return data


def _get_location_coords(location):
    """
    Get the coordinates of the given location.

    Params
    ------
    - location (str): The location to get the coordinates from.

    Returns
    -------
    - tuple: The location and its coordinates. If the location is not found, np.nan is returned.
    """
    geolocator = Nominatim(timeout=50, user_agent="myapplication")
    try:
        geo_location = geolocator.geocode(f"{location}, Myanmar")

        if geo_location:
            return location, (geo_location.latitude, geo_location.longitude)
        else:
            return location, np.nan
    finally:
        # Close the geolocator connection to prevent unnecessary API calls
        del geolocator


def get_coords(df: pd.DataFrame, admin_level: int) -> dict:
    """
    Get the coordinates of the given admin level from the given data using joblib for parallelization.

    Params
    ------
    - df (pd.DataFrame): The data to get the coordinates from.
    - admin_level (int): The admin level to get the coordinates from. Can only be 1, 2, or 3.

    Returns
    -------
    - dict: The coordinates of the given admin level.
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert_df(df)

    location_list = df[f"Admin{admin_level}"].unique()

    results = Parallel(n_jobs=10)(
        delayed(_get_location_coords)(location)
        for location in tqdm(location_list, desc="Getting coordinates")
    )

    location_coords = dict(results)

    assert len(location_coords) > 0, "No coordinates found for the given admin level."
    assert isinstance(location_coords, dict), "Coordinates are not in a dictionary."
    return location_coords


def add_coords_to_df(
    df: pd.DataFrame,
    coords_dict: dict,
    admin_level: int,
) -> pd.DataFrame:
    """
    Add the coordinates of the given admin level to the given data.

    Params
    ------
    - df (pd.DataFrame): The data to add the coordinates to.
    - coords_dict (dict): The coordinates of the given admin level.
    - admin_level (int): The admin level to add the coordinates to. Can only be 1, 2, or 3.

    Returns
    -------
    - pd.DataFrame: The data with the coordinates added.
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."
    assert isinstance(coords_dict, dict), "Coordinates is not a dictionary."

    first_value = next(iter(coords_dict.values()))
    is_valid = isinstance(first_value, tuple) and len(first_value) == 2
    assert is_valid, "Coordinates are not in the correct format."

    df["Latitude"] = df[f"Admin{admin_level}"].apply(
        lambda x: coords_dict.get(x, (None, None))[0]
    )
    df["Longitude"] = df[f"Admin{admin_level}"].apply(
        lambda x: coords_dict.get(x, (None, None))[1]
    )

    assert "Latitude" in df.columns, "Latitude column not found in the DataFrame."
    assert "Longitude" in df.columns, "Longitude column not found in the DataFrame."
    return df


def transform_datetime(df: pd.DataFrame, column: str) -> None:
    """
    Transform the given column in the given data to a datetime format.

    Params
    ------
    - df (pd.DataFrame): The data to transform the column in.
    - column (str): The column to transform to a datetime format.

    """
    assert_df(df)
    assert isinstance(column, str), "Column is not a string."

    df[column] = pd.to_datetime(df[column])
    assert df[column].dtype == "datetime64[ns]", "Column was not transformed correctly."
