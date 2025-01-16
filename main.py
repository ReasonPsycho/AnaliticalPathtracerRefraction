import numpy as np
import matplotlib.pyplot as plt

def lerp(a, b, t):
    return a + (b - a) * t

def sampleTemperature(p):
    seeLevel = 0
    diffrence = 0.002

    t = (p[2] - seeLevel) / diffrence;
    t = np.clip(t, 0, 1);
    return lerp(12.0, 1000.0, t) + 274.15;  # Example temperature field

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

def normalFromPoints(direction, p1, p2):
    """Calculate the normal vector at a point based on temperature gradient."""
    delta = 1  # Small step for finite difference
    # Gradient in the x direction
    t1x = etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2))
    t2x = etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2 + np.array([delta, 0, 0])))
    gradient_x = (t2x - t1x) / delta

    # Gradient in the y direction
    t1y = etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2))
    t2y = etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2 + np.array([0, delta, 0])))
    gradient_y = (t2y - t1y) / delta

    t1z =  etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2))
    t2z = etaFromTemperatures(sampleTemperature(p1),sampleTemperature(p2 + np.array([0, 0, delta])))
    gradient_z = (t2z - t1z) / delta

    if (gradient_z == 0.0):
        return [direction[0], direction[1], direction[2]]

    gradient = np.array([-direction[0] + gradient_x,-direction[1] + gradient_y,-direction[2] + gradient_z])

    return gradient / np.linalg.norm(gradient)

def simulate_ray_path(start_point, initial_direction, steps, delta_step):
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
    path = [[start_point[0],start_point[2]]]
    direction = initial_direction / np.linalg.norm(initial_direction)
    current_point = start_point

    for _ in range(steps):
        next_point = current_point + direction * delta_step
        normal = normalFromPoints(direction,current_point, next_point)
        t1 = sampleTemperature(current_point)
        t2 = sampleTemperature(next_point)
        eta = etaFromTemperatures(t1, t2)
        refracted_dir = refract(direction, normal, eta)

        if refracted_dir is None:
            path.append([next_point[0], next_point[2]])
            # just continue

        direction = refracted_dir / np.linalg.norm(refracted_dir)
        path.append([next_point[0],next_point[2]])
        current_point = next_point

    return np.array(path)

# Initial parameters
start_point = np.array([0.002, 0.0, 0.0])  # Start at origin
initial_direction = np.array([1.0, 0.0, 0.0030])  # Move along +Z direction
delta_step = 0.0001 # Step size

steps = int(6 / delta_step)  # Number of steps to simulate

# Simulate the ray path
ray_paths = []
for i in range(10):
    initial_direction[2] = initial_direction[2] - 0.001
    ray_paths.append(simulate_ray_path(start_point, initial_direction, steps, delta_step))

# Plot the ray path
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for i in range(10):
    ax.plot(ray_paths[i][:, 0], ray_paths[i][:, 1], label='Ray Path', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Simulated Ray Path')
ax.legend()
plt.show()
