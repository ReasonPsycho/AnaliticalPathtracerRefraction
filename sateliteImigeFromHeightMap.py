import requests
from PIL import Image
from io import BytesIO


def fetch_satellite_image(lat1, lng1, lat2, lng2, api_key, image_size=1024):
    """
    Fetch a satellite image from Google Maps Static API.

    :param lat1: Latitude of the first corner.
    :param lng1: Longitude of the first corner.
    :param lat2: Latitude of the second corner.
    :param lng2: Longitude of the second corner.
    :param api_key: Your Google API key.
    :param image_size: Desired image size in pixels (max: 1024x1024).
    :return: PIL.Image object of the satellite image.
    """
    # Calculate center point of the region
    center_lat = (lat1 + lat2) / 2
    center_lng = (lng1 + lng2) / 2

    # Calculate zoom level by approximating the distance
    # Adapt zoom level based on requirements
    zoom_level = 13  # You can dynamically adjust this if needed

    # Construct the Google Static Maps API URL
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lng}"
        f"&zoom={zoom_level}&size={image_size}x{image_size}&maptype=satellite&key={api_key}"
    )

    # Make HTTP request to the API
    response = requests.get(url)

    # Check if the response is valid
    if response.status_code == 200:
        # Convert raw data into an image
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise Exception(f"Error fetching image: {response.status_code}, {response.text}")


# Configuration
LAT1, LNG1 = 42.638618, -8.835686  # First corner latitude and longitude
LAT2, LNG2 = 42.586189, -8.768223  # Second corner latitude and longitude
API_KEY = "TOKEN_HERE"  # Replace with your Google API key
OUTPUT_FILE = "satellite_image.png"  # Output file to save the image

# Fetch and save the satellite image
try:
    satellite_image = fetch_satellite_image(LAT1, LNG1, LAT2, LNG2, API_KEY)
    satellite_image.save(OUTPUT_FILE)  # Save the image to a file
    print(f"Satellite image saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"An error occurred: {e}")