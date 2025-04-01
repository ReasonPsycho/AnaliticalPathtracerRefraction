import numpy as np
from PIL import Image
# Save the texture to a file
import OpenEXR
import Imath
import array

def generate_two_point_temperature_texture(max_altitude_km=10,
                                           point1=(0.0, 30.0),
                                           point2=(1.0, 14.0),
                                           precision=0.1):
    """
    Generates a 1D texture with temperatures defined at two points and smooth transitions back to ISA.

    Parameters:
    - max_altitude_km (float): The maximum altitude (in km) that the texture will cover.
    - point1 (tuple): The first point as (altitude in km, temperature in Fahrenheit).
    - point2 (tuple): The second point as (altitude in km, temperature in Fahrenheit).
    - precision (float): Altitude range in kilometers per pixel in the texture.

    Returns:
    - Image object containing the 1D texture.
    """
    # Constants for ISA
    sea_level_temperature_c = 15.0  # ISA temperature at sea level (Celsius)
    lapse_rate_c_per_km = -6.5  # ISA temperature lapse rate (Celsius per km)
    sea_level_temperature_f = sea_level_temperature_c * 9 / 5 + 32  # Convert ISA sea-level temp to Fahrenheit

    # Calculate the number of pixels based on precision
    num_pixels = int(max_altitude_km / precision)
    max_texture_width = 16384
    num_rows = (num_pixels + max_texture_width - 1) // max_texture_width  # Calculate required rows
    altitudes = np.linspace(0, max_altitude_km, num_pixels)

    # ISA temperature profile in Fahrenheit
    isa_temperatures_f = sea_level_temperature_f + lapse_rate_c_per_km * altitudes * 9 / 5

    # Extract altitude and temperature for two points
    altitude1, temp1 = point1
    altitude2, temp2 = point2

    # Ensure altitude1 < altitude2
    if altitude1 > altitude2:
        altitude1, altitude2 = altitude2, altitude1
        temp1, temp2 = temp2, temp1

    # Linear interpolation between point1 and point2
    interp_mask = np.clip((altitudes - altitude1) / (altitude2 - altitude1), 0, 1)
    interpolated_temps_f = interp_mask * temp2 + (1 - interp_mask) * temp1

    # Smoothly blend into ISA beyond the second point
    # Any altitude below the second point keeps interpolated values
    final_temperatures_f = np.copy(interpolated_temps_f)
    for i, altitude in enumerate(altitudes):
        if altitude > altitude2:
            blend_ratio = (altitude - altitude2) / (max_altitude_km - altitude2)
            final_temperatures_f[i] = (1 - blend_ratio) * interpolated_temps_f[i] + blend_ratio * isa_temperatures_f[i]

    # Normalize temperatures to [0, 255] range
    min_temperature_f = -20.0  # Defines the minimum value to map to 0
    max_temperature_f = 40.0  # Defines the maximum value to map to 255
    normalized_temps = np.clip(
        (final_temperatures_f - min_temperature_f) / (max_temperature_f - min_temperature_f) * 255,
        0, 255
    ).astype(np.uint8)

    # Create 2D texture (RGB, but only red channel is used, with a height of 100 pixels)
    height = 100
    max_texture_width = 16384
    num_rows = (num_pixels + max_texture_width - 1) // max_texture_width  # Rows required to wrap texture
    adjusted_height = height * num_rows  # Final texture height
    texture = np.zeros((adjusted_height, max_texture_width, 3), dtype=np.uint8)
    for row in range(num_rows):
        start_idx = row * max_texture_width
        end_idx = min(start_idx + max_texture_width, num_pixels)
        row_data = normalized_temps[start_idx:end_idx]
        texture[row * height:(row + 1) * height, :len(row_data), 0] = np.tile(row_data, (height, 1))
    # Convert to a PIL Image
    image = Image.fromarray(texture, mode="RGB")
    return image



    # Utility function for saving the texture as PNG
def save_to_png(image, filename):
    """
    Save a PIL Image object as a PNG file.

    Parameters:
    - image (PIL.Image.Image): The image to save.
    - filename (str): The destination PNG file name.
    """
    image.save(filename, format="PNG")
    # Utility function for saving the texture as EXR
def save_to_exr(image, filename):
    """
    Save a PIL Image object as an EXR file using OpenEXR.

    Parameters:
    - image (PIL.Image.Image): The image to save.
    - filename (str): The destination EXR file name.
    """
    # Convert image to NumPy array
    texture_np = np.array(image)

    # Prepare channels (R, G, B)
    r = texture_np[:, :, 0].flatten().astype(np.float32).tobytes()
    g = texture_np[:, :, 1].flatten().astype(np.float32).tobytes()
    b = texture_np[:, :, 2].flatten().astype(np.float32).tobytes()

    # Define resolutions
    height, width, _ = texture_np.shape
    header = OpenEXR.Header(width, height)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                          'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                          'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

    # Save the EXR file
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels({'R': r, 'G': g, 'B': b})
    exr_file.close()
if __name__ == "__main__":
    # Define points with temperature in Fahrenheit at specific altitudes in km
    point1 = (0.000, 12.000 )  # Sea level (0 km), temperature: 65°F
    point2 = (0.100, 2.000 )  # 1 km altitude, temperature: 30°F

    # Generate texture and save with custom precision
    max_altitude_km = 1  # Maximum altitude to cover (in km)
    precision = 0.000001  # Precision: 1 pixel corresponds to 0.1 km of altitude
    texture = generate_two_point_temperature_texture(
        max_altitude_km=max_altitude_km, point1=point1, point2=point2, precision=precision
    )
    print(f"Texture generated with dimensions: {texture.size}")

    # Export texture to EXR format using OpenEXR
    exr_filename = "fig6TemperatureMap.exr"
    save_to_exr(texture, exr_filename)
    # Save the texture to PNG format
    png_filename = "fig6.png"
    save_to_png(texture, png_filename)
    print(f"1D texture saved as '{png_filename}'")
    print(f"1D texture saved as '{exr_filename}'")