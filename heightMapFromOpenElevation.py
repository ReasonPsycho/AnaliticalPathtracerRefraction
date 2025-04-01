import numpy as np
import requests
from PIL import Image


def fetch_elevations_batch(locations, api_key=None, use_google=True, batch_size=256):
    """Fetch elevations in batches for multiple latitude and longitude pairs."""
    elevations = []
    for i in range(0, len(locations), batch_size):
        print(f"Fetching elevations for locations {i+1} to {i+batch_size}...")
        batch = locations[i:i+batch_size]
        if use_google:
            # Google Maps Elevation API
            loc_param = "|".join([f"{lat},{lng}" for lat, lng in batch])
            url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={loc_param}&key={api_key}"
        else:
            # Open Elevation API
            loc_param = "|".join([f"{lat},{lng}" for lat, lng in batch])
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={loc_param}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if use_google:
                elevations.extend([result.get('elevation', 0) for result in data.get('results', [])])
            else:
                elevations.extend([result.get('elevation', 0) for result in data.get('results', [])])
        except (requests.RequestException, ValueError, KeyError):
            elevations.extend([0] * len(batch))  # Placeholder for invalid batches
    return elevations


def generate_height_map(lat1, lng1, lat2, lng2, grid_size, api_key=None, use_google=True):
    """Generate a 2D height map (grid_size x grid_size) within the area defined by two points."""
    lat_spacing = (lat2 - lat1) / (grid_size - 1)
    lng_spacing = (lng2 - lng1) / (grid_size - 1)

    locations = [
        (lat1 + i * lat_spacing, lng1 + j * lng_spacing)
        for i in range(grid_size) for j in range(grid_size)
    ]

    print("Fetching elevations for all locations...")
    elevations = fetch_elevations_batch(locations, api_key, use_google)

    height_map = np.array(elevations).reshape((grid_size, grid_size))
    return height_map


def save_height_map_as_image(height_map, image_size, output_file):
    """Normalize the height map and save it as a grayscale PNG image."""
    # Normalize to 0-255
    normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    normalized = (normalized * 255).astype(np.uint8)

    # Resize to the desired image size
    image = Image.fromarray(normalized)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image.save(output_file)
    print(f"Height map saved to {output_file}")


# Configuration
LAT1, LNG1 =42.609016, -8.800600
  # First corner latitude and longitude
LAT2, LNG2 = 42.578245, -8.758801
  # Second corner latitude and longitude
GRID_SIZE = 400  # 20x20 grid
IMAGE_SIZE = 1024  # Output image size (1024x1024 pixels)
OUTPUT_FILE = "height_map.png"  # Output file name
USE_GOOGLE = True  # Use Google Maps API (True) or Open Elevation API (False)
API_KEY = "TOKEN_HERE" if USE_GOOGLE else None

# Generate and save the height map
height_map = generate_height_map(LAT1, LNG1, LAT2, LNG2, GRID_SIZE, API_KEY, USE_GOOGLE)
save_height_map_as_image(height_map, IMAGE_SIZE, OUTPUT_FILE)
