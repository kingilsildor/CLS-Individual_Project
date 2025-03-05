import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
from config.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.assert_statements import assert_df
from tqdm import tqdm


def _add_features(ax: plt.Axes) -> None:
    """
    Add features to the given axes.

    Params
    ------
    - ax (plt.Axes): The axes to add the features to.
    """
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
    ax.add_feature(cfeature.LAKES, edgecolor="blue", facecolor="none")


def _add_location(lat: int, lon: int, ax: plt.Axes) -> None:
    """
    Add a location to the given axes.

    Params
    ------
    - lat (int): The latitude of the location.
    - lon (int): The longitude of the location.
    - ax (plt.Axes): The axes to add the location to.
    """
    ax.plot(
        lon,
        lat,
        marker="o",
        color="red",
        markersize=2,
        transform=ccrs.PlateCarree(),
    )

def _add_flood_area(lat: int, lon: int, ax: plt.Axes) -> None:
    """
    Add a location to the given axes.

    Params
    ------
    - lat (int): The latitude of the location.
    - lon (int): The longitude of the location.
    - ax (plt.Axes): The axes to add the location to.
    """


def create_map(df: pd.DataFrame, admin_level: int, date: str = "2024-09-09") -> None:
    """
    Create a map of Myanmar with the locations from the given data.

    Params
    ------
    - df (pd.DataFrame): The data to create the map from.
    - admin_level (int): The admin level to create the map from. Can only be 1, 2, or 3.
    - date (str): The date to get the flood data from. Default is "2024-09-09".
    """
    assert admin_level in [1, 2, 3], "Admin level is not 1, 2, or 3."
    assert_df(df)

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
    plt.savefig("results/MyanmarMap.png")
    plt.show()
