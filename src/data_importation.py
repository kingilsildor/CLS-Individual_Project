import glob

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from tqdm import tqdm

from config.config import ROUNDING
from src.assert_statements import assert_df
from src.data_exportation import export_data


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
    dir: str = "../downloaded_files",
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

    data_df = pd.concat(data_frames)
    data_df.sort_values(by=column_sort, ascending=column_asc, inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    assert_df(data_df)
    return data_df


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
    assert_df(df)
    assert isinstance(coords_dict, dict), "Coordinates is not a dictionary."

    first_value = next(iter(coords_dict.values()))
    is_valid = isinstance(first_value, tuple) and len(first_value) == 2
    assert is_valid, "Coordinates are not in the correct format."

    coords = df[f"Admin{admin_level}"].map(coords_dict)
    df[["Latitude", "Longitude"]] = pd.DataFrame(
        coords.apply(lambda x: x if isinstance(x, tuple) else (None, None)).tolist(),
        index=df.index,
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


def create_df(admin_level: int) -> pd.DataFrame:
    """
    Create a DataFrame with the given admin level.
    Group functions that are used together.

    Params
    ------
    - admin_level (int): The admin level to create the DataFrame with.

    Returns
    -------
    - pd.DataFrame: The created DataFrame.
    """
    flood_df = load_data()
    coords_dict = get_coords(flood_df, admin_level)
    flood_df = add_coords_to_df(flood_df, coords_dict, admin_level)
    flood_df = flood_df.dropna()
    flood_df = transform_datetime(flood_df, "Start_Date")
    flood_df = transform_datetime(flood_df, "End_Date")

    export_data(flood_df, "data/flood_data.csv")
    return flood_df


def pivot_df(
    df: pd.DataFrame, admin_level: int, interpolate: bool = True
) -> pd.DataFrame:
    """
    Pivot the given data based on the given admin level and interpolate the data if specified.

    Params
    ------
    - df (pd.DataFrame): The data to pivot.
    - admin_level (int): The admin level to pivot the data for. Can only be 1, 2, or 3.
    - interpolate (bool): Whether to interpolate the data. Default is True.

    Returns
    -------
    - pd.DataFrame: The pivoted data.
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert_df(df)

    # Group by the admin level and end date and sum the flooded area to pivot the data
    pivot_df = (
        df.groupby([f"Admin{admin_level}", "End_Date", "Latitude", "Longitude"])[
            "Flooded_Area_SqKM"
        ]
        .sum()
        .reset_index()
    )
    pivot_df["End_Date"] = pd.to_datetime(pivot_df["End_Date"])
    pivot_df = pivot_df.pivot(
        index=[f"Admin{admin_level}", "Latitude", "Longitude"],
        columns="End_Date",
        values="Flooded_Area_SqKM",
    )

    full_date_range = pd.date_range(
        start=pivot_df.columns.min(), end=pivot_df.columns.max(), freq="D"
    )
    pivot_df = pivot_df.reindex(columns=full_date_range, fill_value=np.nan)

    output_df = pivot_df.copy()
    if interpolate:
        interpolate_df = pivot_df.interpolate(method="linear", axis=1)
        interpolate_df.fillna(0, inplace=True)
        output_df = interpolate_df

    output_df.reset_index(inplace=True)
    output_df.set_index(f"Admin{admin_level}", inplace=True)
    output_df = output_df.round(ROUNDING)
    return output_df


def filter_columns_by_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """
    Filters columns in a DataFrame based on whether they contain the specified month.

    Params
    - df (pd.DataFrame): The input DataFrame.
    - month (str): The month to filter by in 'YYYY-MM' format (e.g., '2024-07').

    Returns
    - pd.DataFrame: A DataFrame with only the columns that contain the specified month.
    """
    base_columns = ["Latitude", "Longitude"]
    month_columns = [col for col in df.columns if month in str(col)]
    filtered_columns = base_columns + month_columns

    return df[filtered_columns]
