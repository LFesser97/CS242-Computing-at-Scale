import numpy as np


def sample_circle_2d(n_samples, radius):
    """
    Sample 'n_samples' points from the circle with center (0,0) and radius 'radius'.
    """
    length = np.sqrt(np.random.uniform(0, radius ** 2, size=n_samples))
    angle = np.pi * np.random.uniform(0, 2, size=n_samples)

    y0 = length * np.cos(angle)
    y1 = length * np.sin(angle)
    points = np.stack([y0, y1], axis=0).T
    
    return points


def sample_triangle_2d(n_samples, side_length):
    """
    Sample 'n_samples' points from the triangle with corners 
    (0,0), ('side_length', 0) and (0, 'side_length').
    """
    r1 = np.random.uniform(0, 1, size=n_samples)
    r2 = np.random.uniform(0, 1, size=n_samples)
    
    y0 = np.sqrt(r1) * (1 - r2)
    y1 = r2 * np.sqrt(r1)
    
    y0 *= side_length
    y1 *= side_length
    points = np.stack([y0, y1], axis=0).T
    
    return points