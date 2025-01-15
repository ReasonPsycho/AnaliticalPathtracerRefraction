import numpy as np
import matplotlib.pyplot as plt

def sampleTemperature(p):
    """Sample temperature at a given point (mock function)."""
    return 300 + 10 * p[0]  # Example temperature field

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
    delta = 0.01  # Small step for finite difference
    t1 = sampleTemperature(p1)
    t2 = sampleTemperature(p2 + np.array([0, 0, delta]))
    gradient_z = (t2 - t1) / delta
    gradient = np.array([0, 0, gradient_z])
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
    path = [start_point]
    direction = initial_direction / np.linalg.norm(initial_direction)
    current_point = start_point

    for _ in range(steps):
        next_point = current_point + direction * delta_step
        normal = normalFromPoints(current_point, next_point)
        t1 = sampleTemperature(current_point)
        t2 = sampleTemperature(next_point)
        eta = etaFromTemperatures(t1, t2)
        refracted_dir = custom_refract(direction, normal, eta)

        if refracted_dir is None:
            break  # Stop simulation if total internal reflection occurs

        direction = refracted_dir / np.linalg.norm(refracted_dir)
        path.append(next_point)
        current_point = next_point

    return np.array(path)

# Initial parameters
start_point = np.array([0.0, 0.0, 0.0])  # Start at origin
initial_direction = np.array([1.0, 0.0, 0.0])  # Move along +Z direction
steps = 100  # Number of steps to simulate
delta_step = 1  # Step size

# Simulate the ray path
ray_path = simulate_ray_path(start_point, initial_direction, steps, delta_step)

# Plot the ray path
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ray_path[:, 0], ray_path[:, 1], ray_path[:, 2], label='Ray Path', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Simulated Ray Path')
ax.legend()
plt.show()
