from datetime import datetime, timedelta
from typing import List, Tuple

import requests
from joblib import Parallel, delayed

# Setup
url_formats = [
    "https://www.moezala.gov.mm/sites/default/files/__MACOSX/Daily%20Waterlevel%20Forecast({})-E_Page_{}.jpg",
    "https://www.moezala.gov.mm/sites/default/files/Daily%20Waterlevel%20Forecast({})-E_Page_{}_0.jpg",
    "https://www.moezala.gov.mm/sites/default/files/Daily%20Waterlevel%20Forecast({})-E_Page_{}.jpg",
]

start_date = datetime(2024, 9, 8)
end_date = datetime(2024, 9, 30)

pages = [1, 2]


def download_image(url: str, filename: str) -> None:
    """
    Download the image from the given URL and save it to the specified filename.

    Params
    ------
    - url (str): The URL of the image to download.
    - filename (str): The name of the file to save the image

    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"✅ Downloaded: {filename}")
    else:
        print(f"❌ Failed: {url}")


def generate_download_tasks() -> List[Tuple[str, str]]:
    """
    Generate the download tasks for the given date range, pages, and URL formats.

    Returns
    -------
    - List[Tuple[str, str]]: The download tasks to execute.
    """
    tasks = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime(
            "%-d-%-m-%Y"
        )  # Format: 8-9-2024 (use %#d-%#m-%Y on Windows)
        for page in pages:
            for url_format in url_formats:
                image_url = url_format.format(date_str, page)
                filename = f"Waterlevel_Forecast_{date_str}_Page_{page}.png"
                tasks.append((image_url, filename))
        current_date += timedelta(days=1)
    return tasks


if __name__ == "__main__":
    tasks = generate_download_tasks()
    Parallel(n_jobs=-1)(
        delayed(download_image)(url, filename) for url, filename in tasks
    )
