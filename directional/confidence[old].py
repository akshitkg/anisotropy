import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point

def generate_random_points_within_polygon(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append([random_point.x, random_point.y])
    return np.array(points)

def calculate_k_function(points, distances, polygon, angle, tolerance):
    k_values = np.zeros(len(distances))
    num_points = len(points)
    if num_points == 0:
        return k_values

    tree = cKDTree(points)
    sector_area = lambda h: (np.pi * h**2 * tolerance) / 180

    for i, dist in enumerate(distances):
        sectors = 0
        for point in points:
            circle = Point(point).buffer(dist)
            sector = Polygon(circle.exterior.coords).intersection(polygon)
            if sector.is_empty:
                continue
            sector_area_val = sector_area(dist)
            points_in_sector = len(tree.query_ball_point(point, dist))
            k_values[i] += points_in_sector / sector_area_val
            sectors += 1
        if sectors > 0:
            k_values[i] /= sectors

    return k_values

def directional_k_function(points, distances, polygon, angles, tolerance, num_simulations):
    k_functions = []
    for angle in angles:
        k_function = calculate_k_function(points, distances, polygon, angle, tolerance)
        k_functions.append(k_function)
    
    simulations = np.zeros((num_simulations, len(distances)))
    for i in range(num_simulations):
        random_points = generate_random_points_within_polygon(polygon, len(points))
        for j, angle in enumerate(angles):
            simulations[i] += calculate_k_function(random_points, distances, polygon, angle, tolerance)
        simulations[i] /= len(angles)

    lower_bound = np.percentile(simulations, 2.5, axis=0)
    upper_bound = np.percentile(simulations, 97.5, axis=0)
    mean_simulation = np.mean(simulations, axis=0)
    
    return k_functions, mean_simulation, lower_bound, upper_bound

def plot_directional_k_function(distances, k_functions, mean_simulation, lower_bound, upper_bound, angles):
    plt.figure(figsize=(14, 8))
    for i, k_function in enumerate(k_functions):
        plt.plot(distances, k_function, label=f'Angle: {angles[i]}°')
    plt.plot(distances, mean_simulation, label='Mean Simulation', linestyle='--', color='gray')
    plt.fill_between(distances, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Distance')
    plt.ylabel('K-function')
    plt.legend()
    plt.title('Directional K-function with Confidence Intervals')
    plt.show()

# Example usage
polygon = Polygon([(0, 0), (12, 0), (12, 12), (0, 12)])  # Replace with actual polygon coordinates
points = np.array([[2, 3], [5, 5], [8, 8], [6, 9], [3, 7], [7, 2]])  # Replace with actual data points
distances = np.linspace(0, 10, 100)
angles = [45, 90, 135]
tolerance = 15
num_simulations = 100

k_functions, mean_simulation, lower_bound, upper_bound = directional_k_function(points, distances, polygon, angles, tolerance, num_simulations)
plot_directional_k_function(distances, k_functions, mean_simulation, lower_bound, upper_bound, angles)
