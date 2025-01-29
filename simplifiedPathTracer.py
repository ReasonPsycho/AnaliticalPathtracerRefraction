import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


def lerp(a, b, t):
    return a + (b - a) * t

def fade(t):
    # Smooth fade function
    return t * t * t * (t * (t * 6 - 15) + 10)

def grad(hash, x, y, z):
    # Gradient function: calculates dot product between a pseudo-random gradient vector and the input vector
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

def cnoise(x, y, z):
    # Permutations table for pseudo-random gradients
    p = np.arange(256, dtype=int)
    np.random.seed(0)  # Use a fixed seed for reproducibility
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    # Unit grid cell coordinates surrounding the input point
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
    Z = int(np.floor(z)) & 255

    # Relative xyz coordinates within the cell
    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)

    # Fade curves for xyz
    u = fade(x)
    v = fade(y)
    w = fade(z)

    # Hash coordinates of the cube corners
    A = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z

    # Add blended results from the 8 corners of the cube
    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                grad(p[BA], x - 1, y, z)),
                        lerp(u, grad(p[AB], x, y - 1, z),
                             grad(p[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                             grad(p[BA + 1], x - 1, y, z - 1)),
                     lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                          grad(p[BB + 1], x - 1, y - 1, z - 1))))

def sampleTemperature(p, texture_1d, precision=0.0001, min_temperature_f=-100.0, max_temperature_f=100.0):
    """
    Sample temperature based on 1D texture, including normalization logic.

    Parameters:
    - p (numpy array): Point in space [x, y, z].
    - texture_1d (numpy array): 1D texture array containing the pixel values (R channel).
    - precision (float): Altitude resolution per pixel (in km).
    - min_temperature_f (float): Minimum temperature mapped to red value 0.
    - max_temperature_f (float): Maximum temperature mapped to red value 255.

    Returns:
    - float: Sampled temperature in degrees Fahrenheit.
    """
    # Constants
    texture_width = len(texture_1d) // 100  # Number of pixels per row in the wrapped 2D texture (10 pixels per row)
    max_height_km = texture_width * 100 * precision  # Total altitude range covered by the 2D texture

    # Map z-coordinate (height in meters) to texture index
    z_km = p[2]  # z (meters) converted to kilometers
    texture_index = z_km / max_height_km * (texture_width * 100)  # Adjust for wrapping logic

    # Handle row and column wrapping logic (wrapped 2D texture)
    row = int(texture_index // 100) % 100
    col = int(texture_index % 100)

    # Retrieve the red value based on wrapping logic
    red_value = texture_1d[row * 100 + col]  # Red component (0–255)

    # Reverse normalization: Convert red_value (0–255) back to temperature (in Fahrenheit)
    temperature_f = min_temperature_f + (red_value / 255.0) * (max_temperature_f - min_temperature_f)

    return temperature_f # + cnoise(p[0], p[1], p[2]) * 0.1

def get_temperature_from_texture(texture, altitude_km, max_altitude_km, precision, min_temperature_f,
                                 max_temperature_f):
    """
    Get the temperature at a specific altitude from a generated 1D temperature texture.

    Parameters:
    - texture (PIL.Image.Image): The generated texture.
    - altitude_km (float): The altitude in kilometers for which to get the temperature.
    - max_altitude_km (float): The maximum altitude covered by the texture (in km).
    - precision (float): The altitude range per pixel in kilometers.
    - min_temperature_f (float): The minimum temperature value mapped to 0 in the texture.
    - max_temperature_f (float): The maximum temperature value mapped to 255 in the texture.

    Returns:
    - temperature (float): The temperature at the given altitude (in Fahrenheit).
    """
    # Convert the texture into a NumPy array
    texture_np = np.array(texture)

    # Calculate the total number of pixels in the 1D data
    texture_width = texture_np.shape[1]
    texture_height = texture_np.shape[0]
    total_pixels = texture_width * (texture_height // 100)

    # Calculate the corresponding pixel index for the given altitude
    pixel_index = min(int(altitude_km / precision), total_pixels - 1)

    # Determine the row and column in the texture where the pixel would be
    row = pixel_index // texture_width
    col = pixel_index % texture_width

    # Get the red channel value at the calculated position (which holds the temperature encoding)
    red_value = texture_np[row * 100, col, 0]

    # Map the normalized value back to the temperature range
    temperature = (red_value / 255) * (max_temperature_f - min_temperature_f) + min_temperature_f

    return temperature

def refract(incident_vec, normal, eta):
    """
    Computes the refracted vector based on an incident vector, surface normal, and eta (n1 / n2).

    Parameters:
    - out (numpy array): Output vector to populate the refracted result.
    - incident_vec (numpy array): Incident vector (3D).
    - normal (numpy array): Normal vector (3D).
    - eta (float): Ratio of refractive indices.

    Returns:
    - None: Updates the out array in place.
    """
    # Ensure vectors are normalized
    incident_vec = incident_vec / np.linalg.norm(incident_vec)
    normal = normal / np.linalg.norm(normal)

    # Dot product between normal and incident_vec
    n_dot_i = np.dot(normal, incident_vec)

    # Snell's Law calculations
    k = 1.0 - eta ** 2 * (1.0 - n_dot_i ** 2)
    if k < 0.0:
        # Total internal reflection - set out to zero vector
        return np.array([0.0, 0.0, 0.0])
    else:
        # Refracted vector calculation
        return eta * incident_vec - (eta * n_dot_i + np.sqrt(k)) * normal

def etaFromTemperatures(t1, t2):
    """Calculate the refractive index ratio based on temperatures."""
    return 1 + 0.000292 * t1 / t2

def normalFromPoints(direction, p1, p2,texture_1d):
    """Calculate the normal vector at a point based on temperature gradient."""
    delta = 0.0001  # Small step for finite difference
    # Gradient in the x direction
    t1x = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2,texture_1d))
    t2x = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2 + np.array([delta, 0, 0]),texture_1d))
    gradient_x = (t2x - t1x) / delta

    # Gradient in the y direction
    t1y = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2,texture_1d))
    t2y = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2 + np.array([0, delta, 0]),texture_1d))
    gradient_y = (t2y - t1y) / delta

    t1z = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2,texture_1d))
    t2z = etaFromTemperatures(sampleTemperature(p1,texture_1d), sampleTemperature(p2 + np.array([0, 0, delta]),texture_1d))
    gradient_z = (t2z - t1z) / delta

    epsilon = 1e-8  # Small value to avoid division by zero
    gradient = np.array([-direction[0] + gradient_x, -direction[1] + gradient_y, -direction[2] + gradient_z])
    gradient_norm = np.linalg.norm(gradient) + epsilon
    return gradient / gradient_norm

def simulate_ray_path(start_point, initial_direction, delta_step,maxX,texture_1d):
    """
    Simulate the ray path across a varying temperature field.

    Parameters:
    - start_point (numpy array): Initial 3D position of the ray
    - initial_direction (numpy array): Initial 3D direction of the ray (normalized)
    - steps (int): Number of steps to simulate
    - delta_step (float): Step size for ray traversal

    Returns:
    - numpy array: Array of ray positions
    """
    path = [[start_point[0], start_point[2]]]
    direction = initial_direction / np.linalg.norm(initial_direction)
    current_point = start_point
    lastDiffrence = 0

    while (math.fabs(current_point[0]) <= maxX and current_point[2]  > 0 ):
        next_point = current_point + direction * delta_step

        t1 = sampleTemperature(current_point,texture_1d)
        t2 = sampleTemperature(next_point,texture_1d)
        currentDifference = math.fabs(t2 - t1) + lastDiffrence


        if (currentDifference > 0.5):
            amount_of_segments = int(currentDifference // 0.8 + 1)
            next_point = current_point + direction * delta_step / amount_of_segments
            t1 = sampleTemperature(current_point, texture_1d)
            t2 = sampleTemperature(next_point, texture_1d)

            for x in range(amount_of_segments - 1):
                eta = etaFromTemperatures(t1, t2)
                normal = normalFromPoints(direction, current_point, next_point, texture_1d)
                direction = refract(direction, normal, eta)
                path.append([next_point[0], next_point[2]])
                current_point = next_point
                next_point = current_point + direction * delta_step / amount_of_segments
                t1 = sampleTemperature(current_point, texture_1d)
                t2 = sampleTemperature(next_point, texture_1d)

        else:
            lastDiffrence =  currentDifference
            path.append([next_point[0], next_point[2]])
            current_point = next_point


    return np.array(path)

def simulate_superior_mirage():
    """
    Fig. 1 is an example of a mirage of this kind: the
    water temperature was between 12 and 14 ºC, that
    of the air between 27 and 30 ºC, the distance to the
    object was 6 km and the camera was 2 m high over
    the sea surface. It was taken in a calm summer day.
    """

    # Initial parameters
    start_point = np.array([0.0, 0.0, 0.002])  # Start at origin
    initial_direction = np.array([1.0, 0.0, 0.010])  # Move along +Z direction

    texture_1d = load_texture("fig1.png")  # Load the wrapped texture based on updated logic

def load_texture(image_path):
    image = Image.open(image_path).convert("RGB")
    texture = np.array(image)
    # Flatten the texture into 1D with wrapping (10 pixels per row)
    return texture[:, :, 0].flatten()

    loops = 10

    # Simulate the ray path
    ray_paths = []
    for i in range(loops):
        actual_direction = initial_direction.copy()
        actual_direction[2] -= 0.001 * i
        delta_step = math.pow(2, 1)  # Step size
        ray_paths.append(simulate_ray_path(start_point, actual_direction, delta_step,6, texture_1d))
        print(i)
    # Plot the ray path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i in range(loops):
        actual_direction = initial_direction.copy()
        actual_direction[2] -= 0.0001 * i
        ax.plot(ray_paths[i][:, 0], ray_paths[i][:, 1], label='Direction: ' +f"{actual_direction[2]:.5f}", marker='o')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    ax.set_title('Simulated Ray Path of Superior Mirage')
    ax.legend()

    ymin = 0.0  # Replace 0.0 with your desired minimum value
    ymax = 0.2  # Replace 0.0 with your desired minimum value
    ax.set_ylim(ymin, None)  # Set minimum Z value and leave maximum value as default (none)

    plt.show()

    # Set the rigid minimum value for Z (y-axis in the diagram)

def simulate_inferior_mirage():
    """
    Fig. 1 is an example of a mirage of this kind: the
    water temperature was between 12 and 14 ºC, that
    of the air between 27 and 30 ºC, the distance to the
    object was 6 km and the camera was 2 m high over
    the sea surface. It was taken in a calm summer day.
    """

    # Initial parameters
    start_point = np.array([0.0, 0.0, 0.002])  # Start at origin
    initial_direction = np.array([1.0, 0.0, 0.000])  # Move along +Z direction

    texture_1d = load_texture("fig6.png")  # Load the wrapped texture based on updated logic

    loops = 10

    # Simulate the ray path
    ray_paths = []
    for i in range(loops):
        actual_direction = initial_direction.copy()
        actual_direction[2] -= 0.0001 * i
        delta_step = math.pow(2.0, 1)  # Step size
        ray_paths.append(simulate_ray_path(start_point, actual_direction, delta_step,8, texture_1d))
        print(i)
    # Plot the ray path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i in range(loops):
        actual_direction = initial_direction.copy()
        actual_direction[2] -= 0.0001 * i
        ax.plot(ray_paths[i][:, 0], ray_paths[i][:, 1], label='Direction: ' +f"{actual_direction[2]:.5f}", marker='o')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    ax.set_title('Simulated Ray Path of Inferior Mirage')
    ax.legend()

    ymin = 0.0  # Replace 0.0 with your desired minimum value
    ymax = 0.2  # Replace 0.0 with your desired minimum value
    ax.set_ylim(ymin, None)  # Set minimum Z value and leave maximum value as default (none)

    plt.show()

    # Set the rigid minimum value for Z (y-axis in the diagram)

#simulate_superior_mirage()
simulate_inferior_mirage()