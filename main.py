import numpy as np
import matplotlib.pyplot as plt

def lerp(a, b, t):
    return a + (b - a) * t

def sampleTemperature(p):
    seeLevel = 0
    diffrence = 0.002

    t = (p[2] - seeLevel) / diffrence;
    t = np.clip(t, 0, 1);
    return lerp(13.0, 12.0, t) + 274.15;  # Example temperature field

def custom_refract(I, N, eta):
    """
    Calculate the refraction vector using Snell's law.

    Parameters:
    - I (numpy array): Incident vector (3D, normalized)
    - N (numpy array): Surface normal vector (3D, normalized)
    - eta (float): Index of refraction ratio (n1 / n2)

    Returns:
    - numpy array: Refracted vector (3D) or None if total internal reflection occurs
    """
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)
    cosThetaI = -np.dot(I, N)
    k = 1.0 - eta**2 * (1.0 - cosThetaI**2)
    if k < 0.0:
        return None  # Total internal reflection
    return eta * I + (eta * cosThetaI - np.sqrt(k)) * N

def etaFromTemperatures(t1, t2):
    """Calculate the refractive index ratio based on temperatures."""
    return 1 + 0.000292 * t1 / t2

def normalFromPoints(p1, p2):
    """Calculate the normal vector at a point based on temperature gradient."""
    delta = 1  # Small step for finite difference
    # Gradient in the x direction
    t1x = sampleTemperature(p1)
    t2x = sampleTemperature(p2 + np.array([delta, 0, 0]))
    gradient_x = (t2x - t1x) / delta

    # Gradient in the y direction
    t1y = sampleTemperature(p1)
    t2y = sampleTemperature(p2 + np.array([0, delta, 0]))
    gradient_y = (t2y - t1y) / delta

    t1z = sampleTemperature(p1)
    t2z = sampleTemperature(p2 + np.array([0, 0, delta]))
    gradient_z = (t2z - t1z) / delta
    gradient = np.array([gradient_x,gradient_y, gradient_z])
    if(gradient_z == 0.0):
        return [0.0,0.0,0.0]
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
        normal = normalFromPoints(current_point, next_point)
        t1 = sampleTemperature(current_point)
        t2 = sampleTemperature(next_point)
        eta = etaFromTemperatures(t1, t2)
        refracted_dir = custom_refract(direction,-direction + normal, eta)

        if refracted_dir is None:
            path.append([next_point[0], next_point[2]])
            # just continue

        direction = refracted_dir / np.linalg.norm(refracted_dir)
        path.append([next_point[0],next_point[2]])
        current_point = next_point

    return np.array(path)

# Initial parameters
start_point = np.array([0.0, 0.0, 0.0])  # Start at origin
initial_direction = np.array([1.0, 0.0, 0.0005])  # Move along +Z direction
delta_step = 0.5  # Step size

steps = int(6 / delta_step)  # Number of steps to simulate

# Simulate the ray path
ray_paths = []
for i in range(10):
    initial_direction[2] = initial_direction[2] - 0.0001
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
