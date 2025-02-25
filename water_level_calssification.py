import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from geopy.geocoders import Nominatim
from tqdm import tqdm

def load_data(
    dir: str = "../downloaded_files",
    column_sort: list = ["Admin2", "Start_Date", "End_Date", "Flooded_Area_SqKM"],
    column_asc: list = [True, True, True, False],
) -> pd.DataFrame:
    """
    Load data from the given directory and sort it based on the given columns.

    Params
    ------
    dir (str): The directory where the data is stored. Default is "../downloaded_files".
    column_sort (list): The columns to sort the data by. Default is ["Admin2", "Start_Date", "End_Date", "Flooded_Area_SqKM"].
    column_asc (list): The order of the columns to sort the data by. Default is [True, True, True, False].

    Returns
    -------
    pd.DataFrame: The sorted data.
    """
    assert isinstance(dir, str), "Directory is not a string."
    assert isinstance(column_sort, list), "column_sort is not a list."
    assert isinstance(column_asc, list), "column_asc is not a list."

    files = glob.glob(f"{dir}/*.csv")
    assert len(files) > 0, "No files found in the given directory."
    data = pd.concat([pd.read_csv(file) for file in files])

    data.sort_values(by=column_sort, ascending=column_asc, inplace=True)
    data.reset_index(drop=True, inplace=True)

    assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."
    assert data.shape[0] > 0, "No data found in the given directory."
    assert data.shape[1] > 0, "No columns found in the given directory."
    return data

def get_coords(df: pd.DataFrame, admin_level: int) -> dict:
    """
    Get the coordinates of the given admin level from the given data.

    Params
    ------
    df (pd.DataFrame): The data to get the coordinates from.
    admin_level (str): The admin level to get the coordinates from. Can only be 1, 2, or 3.

    Returns
    -------
    dict: The coordinates of the given admin level.
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."

    geolocator = Nominatim(timeout=500, user_agent="myapplication")
    assert geolocator, "Geolocator is not defined."

    location_coords = {}
    location_list = df[f"Admin{admin_level}"].unique()

    for location in location_list:
        geo_location = geolocator.geocode(f"{location}, Myanmar")
        if geo_location:
            location_coords[location] = (geo_location.latitude, geo_location.longitude)
        else:
            location_coords[location] = np.nan
    
    assert len(location_coords) > 0, "No coordinates found for the given admin level."
    print(f"{len(location_coords)} coordinates for Admin{admin_level} have been found.")
    return location_coords