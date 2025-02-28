import datetime
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from tqdm import tqdm


def assert_df(df: pd.DataFrame) -> None:
    """
    Assert that the given data is a pandas DataFrame and has rows and columns.

    Params
    ------
    - df (pd.DataFrame): The data to assert.
    """
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."


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
    - df (pd.DataFrame): The data to get the coordinates from.
    - admin_level (int): The admin level to get the coordinates from. Can only be 1, 2, or 3.

    Returns
    -------
    - dict: The coordinates of the given admin level.
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


def create_map(df: pd.DataFrame, admin_level: int) -> None:
    """
    Create a map of Myanmar with the locations from the given data.

    Params
    ------
    - df (pd.DataFrame): The data to create the map from.
    - admin_level (int): The admin level to create the map from. Can only be 1, 2, or 3.
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
    assert_df(df)

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
    - df (pd.DataFrame): The data to export.
    - file_name (str): The name of the file to export the data to.
    """
    assert_df(df)
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
    - df (pd.DataFrame): The data to transform the column in.
    - column (str): The column to transform to a datetime format.

    Returns
    -------
    - pd.DataFrame: The data with the column transformed to a datetime format.
    """
    assert_df(df)
    assert isinstance(column, str), "Column is not a string."

    df[column] = pd.to_datetime(df[column])
    assert df[column].dtype == "datetime64[ns]", "Column was not transformed correctly."
    return df


def create_flood_level(df: pd.DataFrame, nbins: int):
    assert_df(df)

    df = df.copy()
    max_flood_area = df["Flooded_Area_SqKM"].max()

    flood_bins = np.linspace(0, max_flood_area, nbins + 1)

    flood_labels = [i for i in range(1, nbins + 1)]
    df["Flooded_Category"] = pd.cut(
        df["Flooded_Area_SqKM"], bins=flood_bins, labels=flood_labels
    )

    assert "Flooded_Category" in df.columns, (
        "Flooded_Category column not found in the DataFrame."
    )
    return df


def flood_analysis(
    df: pd.DataFrame, admin_level: int, n_jobs: int = -1
) -> pd.DataFrame:
    """
    Perform flood analysis in parallel for the given data, admin level, location, and number of bins.

    Params
    ------
    - df (pd.DataFrame): The data to perform the flood analysis on.
    - admin_level (int): The admin level to perform the flood analysis on. Can only be 1, 2, or 3.
    - n_jobs (int): The number of jobs to use for the flood analysis. Default is -1.

    Returns
    -------
    - pd.DataFrame: The results of the flood analysis.
    """

    def _process_date(
        df: pd.DataFrame,
        date: datetime.date,
        start_date: datetime.date,
        admin_level: int,
        location: str,
    ) -> pd.DataFrame:
        """
        Process the given date for the flood analysis.

        Params
        ------
        df (pd.DataFrame): The data to process.
        date (datetime): The date to process.
        start_date (datetime): The start date of the data.
        admin_level (int): The admin level to process.
        location (str): The location to process.

        Returns
        -------
        pd.DataFrame: The results of the flood analysis for the given date.
        """
        assert_df(df)

        days_passed = (date - start_date).days
        df_temp = df[df["End_Date"].dt.date == date]

        # Get the mode of the flood category
        flood_category_df = (
            df_temp.groupby(f"Admin{admin_level}")["Flooded_Category"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .reset_index()
        )
        flood_category_mode = flood_category_df["Flooded_Category"].iloc[0]

        return pd.DataFrame({"#Days": [days_passed], location: [flood_category_mode]})

    def _collect_results(
        df: pd.DataFrame, admin_level: int, location: str, n_jobs: int
    ) -> pd.DataFrame:
        """
        Collect the results of the flood analysis for the given data, admin level, location, and number of bins.

        Params
        ------
        - df (pd.DataFrame): The data to collect the results from.
        - admin_level (int): The admin level to collect the results from. Can only be 1, 2, or 3.
        - location (str): The location to collect the results from.
        - n_jobs (int): The number of jobs to use for the flood analysis.

        Returns
        -------
        - pd.DataFrame: The results of the flood analysis.
        """
        assert_df(df)

        start_date = df["Start_Date"].dt.date.min()
        unique_end_dates = df["End_Date"].dt.date.unique()

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_date)(df, date, start_date, admin_level, location)
            for date in unique_end_dates
        )

        category_df = pd.concat(results, ignore_index=True)
        return category_df

    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert_df(df)
    locations = df[f"Admin{admin_level}"].unique()
    flee_df = pd.DataFrame(columns=["#Days"])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_collect_results)(df, admin_level, location, n_jobs)
        for location in tqdm(locations, desc="Categorizing floods")
    )

    for category_df in results:
        flee_df = flee_df.merge(category_df, on="#Days", how="outer")
    return flee_df


def main():
    # # Setup the data
    # flood_df = load_data()
    # coords_dict = get_coords(flood_df, 2)
    # flood_df = add_coords_to_df(flood_df, coords_dict, 2)
    # flood_df = transform_datetime(flood_df, "Start_Date")
    # flood_df = transform_datetime(flood_df, "End_Date")

    # flood_df = categorize_floods(flood_df, 4)
    # export_data(flood_df, "flood_data.csv")

    df = pd.read_csv("flood_data.csv")
    df = create_flood_level(df, nbins)
    df["Start_Date"] = pd.to_datetime(df["Start_Date"])
    df["End_Date"] = pd.to_datetime(df["End_Date"])

    data = flood_analysis(df, admin_level=2)
    export_data(data, "flood_analysis.csv")
    print(data)


if __name__ == "__main__":
    main()
