import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
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

    def _load_and_process_csv(file):
        df = pd.read_csv(file)
        if df is not None:
            return df
        else:
            return pd.DataFrame()

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


def get_coords(df: pd.DataFrame, admin_level: int) -> dict:
    """
    Get the coordinates of the given admin level from the given data using joblib for parallelization.

    Params
    ------
    df (pd.DataFrame): The data to get the coordinates from.
    admin_level (int): The admin level to get the coordinates from. Can only be 1, 2, or 3.

    Returns
    -------
    dict: The coordinates of the given admin level.
    """

    def _get_location_coords(location):
        geolocator = Nominatim(timeout=50, user_agent="myapplication")
        try:
            geo_location = geolocator.geocode(f"{location}, Myanmar")

            if geo_location:
                return location, (geo_location.latitude, geo_location.longitude)
            else:
                return location, np.nan
        finally:
            del geolocator

    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."

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
    df (pd.DataFrame): The data to add the coordinates to.
    coords_dict (dict): The coordinates of the given admin level.
    admin_level (int): The admin level to add the coordinates to. Can only be 1, 2, or 3.

    Returns
    -------
    pd.DataFrame: The data with the coordinates added.
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


def create_map(df: pd.DataFrame, admin_level: int) -> None:
    """
    Create a map of Myanmar with the locations from the given data.

    Params
    ------
    df (pd.DataFrame): The data to create the map from.
    admin_level (int): The admin level to create the map from. Can only be 1, 2, or 3.
    """

    def _add_features(ax: plt.Axes) -> None:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
        ax.add_feature(cfeature.LAKES, edgecolor="blue", facecolor="none")

    def _add_location(lat: int, lon: int, ax: plt.Axes) -> None:
        ax.plot(
            lon,
            lat,
            marker="o",
            color="red",
            markersize=2,
            transform=ccrs.PlateCarree(),
        )

    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."

    # Define the map boundaries for Myanmar
    LON_MIN, LON_MAX = 92, 102
    LAT_MIN, LAT_MAX = 9, 29

    _, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    _add_features(ax)

    locations = df[f"Admin{admin_level}"].unique()
    coordinates = df[df[f"Admin{admin_level}"].isin(locations)][
        ["Latitude", "Longitude"]
    ]
    coordinates = coordinates.drop_duplicates()

    for lat, lon in tqdm(
        zip(coordinates["Latitude"], coordinates["Longitude"]), desc="Adding locations"
    ):
        _add_location(lat, lon, ax)

    plt.title(f"Locations in Myanmar from the dataset for Admin{admin_level}")
    plt.tight_layout()
    plt.savefig("map.png")
    plt.show()


def export_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Export the given data to a CSV file.

    Params
    ------
    df (pd.DataFrame): The data to export.
    file_name (str): The name of the file to export the data to.
    """
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."
    assert isinstance(file_name, str), "File name is not a string."

    df.to_csv(file_name, index=False)
    assert pd.read_csv(file_name).shape == df.shape, (
        "Data was not exported to the file correctly."
    )


def transform_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Transform the given column in the given data to a datetime format.

    Params
    ------
    df (pd.DataFrame): The data to transform the column in.
    column (str): The column to transform to a datetime format.

    Returns
    -------
    pd.DataFrame: The data with the column transformed to a datetime format.
    """
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."
    assert isinstance(column, str), "Column is not a string."

    df[column] = pd.to_datetime(df[column])
    assert df[column].dtype == "datetime64[ns]", "Column was not transformed correctly."
    return df


def categorize_floods(df: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """
    Categorize the floods in the given data into bins based on the given bin size.

    Params
    ------
    df (pd.DataFrame): The data to categorize the floods in.
    bin_size (int): The size of the bins to categorize the floods into.

    Returns
    -------
    pd.DataFrame: The data with the floods categorized into bins.
    """
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."

    # Create flood bins
    max_flood = df["Flooded_Area_SqKM"].max()
    flood_bins = np.linspace(0, max_flood, bin_size).tolist()
    flood_labels = [f"F{i}" for i in range(1, bin_size)]

    df["Flooded_Area_Category"] = pd.cut(
        df["Flooded_Area_SqKM"], bins=flood_bins, labels=flood_labels
    )
    assert "Flooded_Area_Category" in df.columns, (
        "Flooded_Area_Category not found in the DataFrame."
    )
    return df


def main():
    # Setup the data
    flood_df = load_data()
    coords_dict = get_coords(flood_df, 2)
    flood_df = add_coords_to_df(flood_df, coords_dict, 2)
    flood_df = transform_datetime(flood_df, "Start_Date")
    flood_df = transform_datetime(flood_df, "End_Date")

    flood_df = categorize_floods(flood_df, 4)
    export_data(flood_df, "flood_data.csv")


if __name__ == "__main__":
    main()
