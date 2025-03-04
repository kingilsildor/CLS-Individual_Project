import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.assert_statements import assert_df
from src.data_exportation import export_data
from tqdm import tqdm


def create_flood_level(
    df: pd.DataFrame, admin_level: int, nbins: int = 4
) -> pd.DataFrame:
    """
    Algorithm to categorize the flood levels into the given number of bins.

    Params
    ------
    - df (pd.DataFrame): The data to categorize the flood levels in.
    - admin_level (int): The admin level to categorize the flood levels in. Can only be 1, 2, or 3.
    - nbins (int): The number of bins to categorize the flood levels into. Default is 4.

    Returns
    -------
    - pd.DataFrame: The data with the flood levels categorized.
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
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


def interpolate_flood_level(df: pd.DataFrame) -> pd.DataFrame: ...


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
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert_df(df)
    locations = df[f"Admin{admin_level}"].unique()
    data_df = pd.DataFrame(columns=["#Days"])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_collect_results)(df, admin_level, location, n_jobs)
        for location in tqdm(locations, desc="Categorizing floods")
    )

    for category_df in results:
        data_df = data_df.merge(category_df, on="#Days", how="outer")
    export_data(data_df, "data/flood_analysis.csv")
    return data_df
