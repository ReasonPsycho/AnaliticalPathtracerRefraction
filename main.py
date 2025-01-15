import numpy as np
import matplotlib.pyplot as plt

def sampleTemperature(p):
    return 1;

def custom_refract(I, N, eta):
    """
    Calculate the refraction vector using Snell's law.

    Parameters:
    - I (numpy array): Incident vector (3D, normalized)
    - N (numpy array): Surface normal vector (3D, normalized)
    - eta (float): Index of refraction ratio (n1 / n2)

    Returns:
    - numpy array: Refracted vector (3D) or a zero vector if total internal reflection occurs
    """
    # Normalize the input vectors
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)

    # Calculate the cosine of the angle of incidence
    cosThetaI = -np.dot(I, N)

    # Compute the discriminant (k)
    k = 1.0 - eta**2 * (1.0 - cosThetaI**2)

    # If k is negative, total internal reflection occurs
    if k < 0.0:
        return np.array([0.0, 0.0, 0.0])  # Return zero vector

    # Compute the refracted vector
    refracted = eta * I + (eta * cosThetaI - np.sqrt(k)) * N
    return refracted

def etaFromTemperatures(t1, t2):
    return 1 + 0.000292*t1/t2

def normalFromPoints(p1, p2):
        """
        Calculate the temperature normal at the point defined by p1 and p2.

        Args:
            ptc: Path tracing context (user-defined, required for sampling and IOR functions).
            p1: A 3D point (numpy array or list of floats).
            p2: A 3D point (numpy array or list of floats).

        Returns:
            A normalized 3D gradient vector (numpy array).
        """
        delta = np.linalg.norm(np.array(p1) - np.array(p2))  # Calculate the distance between p1 and p2

        # Sample the refractive index at nearby points for finite difference computation

        eta_z1 = etaFromTemperatures(sampleTemperature(p1), sampleTemperature(p2))
        eta_z2 = etaFromTemperatures(
            etaFromTemperatures(p1),
            etaFromTemperatures(p2 + np.array([0, 0, delta]))
        )

        # Compute finite differences (gradients)
        gradient_z = (eta_z2 - eta_z1) / delta  # Approximate gradient in z direction

        # Combine to form the gradient vector (normal vector to the isosurface)
        gradient = np.array([0, 0, gradient_z])
        return gradient / np.linalg.norm(gradient)  # Normalize the gradient vector


# Materials with indices of refraction (air -> glass -> water)
indices_of_refraction = [1.0, 1.5, 1.33, 1.0]
incident_angle = 45  # Initial angle of incidence

# Simulate the ray path
angles = [incident_angle]
for i in range(len(indices_of_refraction) - 1):
    n1 = indices_of_refraction[i]
    n2 = indices_of_refraction[i + 1]
    theta1 = angles[-1]
    theta2 = custom_refract(n1, n2, theta1)
    if theta2 is None:
        break  # Total internal reflection; stop the simulation
    angles.append(theta2)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(range(len(angles)), angles, marker='o', label='Refracted Angles')
plt.xlabel('Interaction Index')
plt.ylabel('Angle (degrees)')
plt.title('Refraction of Ray Across Material Boundaries')
plt.grid(True)
plt.legend()
plt.show()
