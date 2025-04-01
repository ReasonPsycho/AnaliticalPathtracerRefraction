import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from fontTools.misc.textTools import tostr

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

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def sampleTemperature(p, temperatureP1, temperatureP2):
    temp1, altitude1 = temperatureP1  # Temperature and altitude of first point
    temp2, altitude2 = temperatureP2  # Temperature and altitude of second point
    difference = abs(altitude1 - altitude2)  # Altitude difference (range for blending)

    # Below lower altitude range
    if p[2] < altitude1:
        return temp1 + 274.15  # Return temperature in Kelvin

    # Linear interpolation between temp1 and temp2 within range altitude1 to altitude2
    elif p[2] < altitude2:
        t = clamp((p[2] - altitude1) / difference, 0.0, 1.0)  # Correctly calculate t for blending
        return lerp(temp1, temp2, t) + 274.15

    # Blend with ISA model above the upper altitude but within blending range
    elif p[2] < altitude2 + difference:
        isa_temperature = 288.15 - (6.5 * p[2])  # Standard ISA temperature model
        t = clamp((p[2] - altitude2) / difference, 0.0, 1.0)  # Blending ratio
        return lerp(temp2, isa_temperature, t) + 274.15

    # Fully outside range, use ISA model
    return 288.15 - (6.5 * p[2])

def refract(incident_vec, normal, eta):
    """
    Computes the refracted vector based on an incident vector, surface normal, and eta (n1 / n2).

    Parameters:
    - incident_vec (numpy array): Incident vector (3D).
    - normal (numpy array): Normal vector (3D).
    - eta (float): Ratio of refractive indices.

    Returns:
    - numpy array: Refracted vector (3D), or a zero vector in case of total internal reflection
      or invalid inputs (e.g., zero vectors).
    """
    # Avoid division by zero by checking norm before normalization
    incident_norm = np.linalg.norm(incident_vec)
    normal_norm = np.linalg.norm(normal)

    if incident_norm == 0 or normal_norm == 0:
        # Invalid input, return zero vector
        return np.array([0.0, 0.0, 0.0])

    # Normalize vectors
    incident_vec = incident_vec / incident_norm
    normal = normal / normal_norm

    # Dot product between normal and incident_vec
    n_dot_i = np.dot(normal, incident_vec)

    # Snell's Law calculations
    k = 1.0 - eta ** 2 * (1.0 - n_dot_i ** 2)
    if k < 0.0:
        # Total internal reflection - return zero vector
        return np.array([0.0, 0.0, 0.0])
    else:
        # Refracted vector calculation
        return eta * incident_vec - (eta * n_dot_i + np.sqrt(k)) * normal

def etaFromTemperatures(t1, t2):
    """Calculate the refractive index ratio based on temperatures."""
    return 1 + 0.000292 * t1 / t2

def normalFromPoints(direction, p1, p2,temperatureP1,temperatureP2):
    """Calculate the normal vector at a point based on temperature gradient."""
    delta = 0.00001  # Small step for finite difference
    # Gradient in the x direction
    t1x = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2), sampleTemperature(p2, temperatureP1, temperatureP2))
    t2x = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2),
                              sampleTemperature(p2 + np.array([delta, 0, 0]), temperatureP1, temperatureP2))
    gradient_x = (t2x - t1x) / delta

    # Gradient in the y direction
    t1y = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2), sampleTemperature(p2, temperatureP1, temperatureP2))
    t2y = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2),
                              sampleTemperature(p2 + np.array([0, delta, 0]), temperatureP1, temperatureP2))
    gradient_y = (t2y - t1y) / delta

    t1z = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2), sampleTemperature(p2, temperatureP1, temperatureP2))
    t2z = etaFromTemperatures(sampleTemperature(p1, temperatureP1, temperatureP2),
                              sampleTemperature(p2 + np.array([0, 0, delta]), temperatureP1, temperatureP2))
    gradient_z = (t2z - t1z) / delta

    normalMultiplayer = (math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1]) + math.fabs(p1[2] - p2[2])) * 1000

    epsilon = 1e-8  # Small value to avoid division by zero
    gradient = np.array([-direction[0] + gradient_x * normalMultiplayer, -direction[1] + gradient_y* normalMultiplayer, -direction[2] + gradient_z* normalMultiplayer])
    gradient_norm = np.linalg.norm(gradient) + epsilon
    return gradient / gradient_norm

def rk4_step(position, direction, delta_t, temperatureP1,temperatureP2):
    """
    Perform a single RK4 step for the ray position and direction using the refractive field.

    Parameters:
    - position (numpy array): Current position vector (x, y, z).
    - direction (numpy array): Current direction vector (dx, dy, dz).
    - delta_t (float): Time step for RK4.
    - texture_1d (numpy array): Temperature field, used for refractive index gradients.

    Returns:
    - new_position (numpy array): Updated position vector after delta_t.
    - new_direction (numpy array): Updated direction vector after delta_t.
    """

    def compute_derivatives(pos, dir):
        """
        Compute position and direction derivatives for RK4.
        - dP/dt = dir (current direction)
        - dD/dt = gradient of refractive index affecting the direction
        """
        # Position derivative is simply the direction
        position_derivative = dir

        # Compute the refractive gradient and find the effect on the direction
        next_point = pos + dir * delta_t
        temp1 = sampleTemperature(pos, temperatureP1, temperatureP2)
        temp2 = sampleTemperature(next_point, temperatureP1, temperatureP2)
        eta = etaFromTemperatures(temp1, temp2)

        # Compute the normal from refractive gradients
        normal = normalFromPoints(dir, pos, next_point,temperatureP1,temperatureP2)

        # Compute refracted direction
        direction_derivative = refract(dir, normal, eta) - dir  # Change in direction

        return position_derivative, direction_derivative

    # RK4 integration steps
    k1_pos, k1_dir = compute_derivatives(position, direction)
    k2_pos, k2_dir = compute_derivatives(
        position + 0.5 * delta_t * k1_pos,
        direction + 0.5 * delta_t * k1_dir
    )
    k3_pos, k3_dir = compute_derivatives(
        position + 0.5 * delta_t * k2_pos,
        direction + 0.5 * delta_t * k2_dir
    )
    k4_pos, k4_dir = compute_derivatives(
        position + delta_t * k3_pos,
        direction + delta_t * k3_dir
    )

    # Combine weighted derivatives to compute the change over delta_t
    new_position = position + (delta_t / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
    new_direction = direction + (delta_t / 6.0) * (k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir)

    return new_position, new_direction

def simulate_ray_path(start_point, initial_direction, delta_step, maxX, temperatureP1,temperatureP2):
    """
    Simulate the ray path using RK4 integration for accurate refraction calculations.
    """
    path = [[start_point[0], start_point[2]]]
    position = start_point
    if(initial_direction[2] == 0):
        return  np.array(path)

    direction = initial_direction / np.linalg.norm(initial_direction)

    while math.fabs(position[0]) <= maxX and position[2] > 0 and len(path) < 20:
        # Perform a single RK4 step
        position, direction = rk4_step(position, direction, delta_step, temperatureP1,temperatureP2)

        # Store the path (projecting to X-Z plane for simplicity)
        path.append([position[0], position[2]])

    return np.array(path)

def load_texture(image_path):
    image = Image.open(image_path).convert("RGB")
    texture = np.array(image)
    # Flatten the texture into 1D with wrapping (10 pixels per row)
    return texture[:, :, 0].flatten()

def calculate_initial_direction(min_direction,max_direction,current_loop,max_loops):
    return lerp(min_direction,max_direction,current_loop/max_loops)

loops = 100
ymin = 0.0  # Replace 0.0 with your desired minimum value
ymax = 0.01  # Replace 0.0 with your desired minimum value
xmin = 0.0  # Replace 0.0 with your desired minimum value
xmax = 8  # Replace 0.0 with your desired minimum value
start_point = np.array([0.0, 0.0, 0.002])  # Start at origin
min_direction = np.array([1.0, 0.0, -0.002])  # Start at origin
max_direction = np.array([1.0, 0.0, 0.002])  # Start at origin
delta_step = math.pow(2, 1)  # Step size

def simulate_superior_mirage(temperatureP1,temperatureP2):
    """
    Fig. 1 is an example of a mirage of this kind: the
    water temperature was between 12 and 14 ºC, that
    of the air between 27 and 30 ºC, the distance to the
    object was 6 km and the camera was 2 m high over
    the sea surface. It was taken in a calm summer day.
    """

    # Initial parameters

    texture_1d = load_texture("fig1.png")  # Load the wrapped texture based on updated logic


    # Simulate the ray path
    ray_paths = []
    for i in range(loops):
        ray_paths.append(simulate_ray_path(start_point, calculate_initial_direction(min_direction,max_direction,i,loops), delta_step, 6, temperatureP1,temperatureP2))
        print(i)
    # Plot the ray path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i in range(loops):
        ax.plot(ray_paths[i][:, 0], ray_paths[i][:, 1], label='Direction: ' + f"{calculate_initial_direction(min_direction,max_direction,i,loops)[2]:.5f}", marker='o')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    ax.set_title('Symulowane ścieżki promieni mirażu górnego')
    #ax.legend()


    ax.set_ylim(ymin, ymax)  # Set minimum Z value and leave maximum value as default (none)
    ax.set_xlim(xmin, 6)  # Set minimum Z value and leave maximum value as default (none)

    plt.show()


    # Set the rigid minimum value for Z (y-axis in the diagram)

def simulate_inferior_mirage(temperatureP1,temperatureP2):
    """
    Fig. 1 is an example of a mirage of this kind: the
    water temperature was between 12 and 14 ºC, that
    of the air between 27 and 30 ºC, the distance to the
    object was 6 km and the camera was 2 m high over
    the sea surface. It was taken in a calm summer day.
    """

    texture_1d = load_texture("fig6.png")  # Load the wrapped texture based on updated logic
    # Simulate the ray path
    ray_paths = []
    for i in range(loops):
        ray_paths.append(simulate_ray_path(start_point, calculate_initial_direction(min_direction,max_direction,i,loops), delta_step,8, temperatureP1,temperatureP2))
        print(i)
    # Plot the ray path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i in range(loops):

        ax.plot(ray_paths[i][:, 0], ray_paths[i][:, 1], label='Direction: ' +f"{calculate_initial_direction(min_direction,max_direction,i,loops)[2]:.5f}", marker='o')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    ax.set_title('Symulowane ścieżki promieni mirażu dolnego')
    #ax.legend()

    ax.set_ylim(ymin, ymax)  # Set minimum Z value and leave maximum value as default (none)
    ax.set_xlim(xmin, xmax)  # Set minimum Z value and leave maximum value as default (none)

    plt.show()

    # Set the rigid minimum value for Z (y-axis in the diagram)

simulate_superior_mirage((13,0),(29,0.04))
simulate_inferior_mirage((10,0),(6,0.04))

