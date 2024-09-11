import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
import random
    
def generate(min_val, max_val, points_per_process):
    points = []
    for _ in range(points_per_process):
        point = [random.uniform(min_val[0], max_val[0]),
                 random.uniform(min_val[1], max_val[1]),
                 random.uniform(min_val[2], max_val[2])]
        points.append(point)
    return points

def point_inside_prism(point, prism):
    # Check if the point is inside the prism
    x, y, z = point
    x_vals = [vertex[0] for vertex in prism]
    y_vals = [vertex[1] for vertex in prism]
    z_vals = [vertex[2] for vertex in prism]
        
    # Check if the point is within the prism's boundaries
    if min(x_vals) <= x <= max(x_vals) and min(y_vals) <= y <= max(y_vals) and min(z_vals) <= z <= max(z_vals):
        return True
    else:
        return False

def pre_processing():
    # initialize ARRAY of vertices
    vertices = []
    prisms = []

    # READ test case
    with open('prisms.txt') as file:
        # use MAP to iterate over each line
        count, deg_acc = map(float, file.readline().split())
        for _ in range(int(count)):
            x1, y1, z1, x2, y2, z2 = map(float, file.readline().split())
            two_vertices = np.array([x1, y1, z1, x2, y2, z2])
            prism = np.array([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1],
                              [x1, y2, z1], [x1, y1, z2], [x2, y1, z2],
                              [x2, y2, z2], [x1, y2, z2]])
            vertices.append(two_vertices)
            prisms.append(prism)

    # CREATE box with the min and max points
    points = np.concatenate(prisms)
    min_val = np.min(points, axis=0)
    max_val = np.max(points, axis=0)
    return min_val, max_val, prisms, deg_acc

def mapper(min_val, max_val, points_per_process, prism):
    points = generate(min_val, max_val, points_per_process)
    points_inside_prism = sum(point_inside_prism(point, prism) for point in points)
    return points_inside_prism

def collecting(min_val, max_val, prisms, deg_acc):
    num_processes = 150
    points_per_process = 10000
    pool = mp.Pool(processes=num_processes)
    results = [pool.apply_async(mapper, args=(min_val, max_val, points_per_process, prism)) for prism in prisms]
    return results, deg_acc

def reducer(results, min_val, max_val, deg_acc):
    # if INSIDE prisms, count the point
    points_inside_prisms = 0
    for result in results:
        points_inside_prisms += result.get()

    # volume of bounding box
    x = max_val[0] - min_val[0]
    y = max_val[1] - min_val[1]
    z = max_val[2] - min_val[2]
    box_volume = x * y * z
        
    # volume as ratio of points inside pancakes to points generated multiplied by volume of bounding box
    denom = float(150 * 10000) * box_volume
    volume = points_inside_prisms / denom
    volume = (volume / deg_acc) * deg_acc

    print(f"Points inside: {points_inside_prisms} \nTotal points: {150 * 10000} \nBoundary box volume: {box_volume:.3f} \nVolume as ratio of points inside pancakes to points generated multiplied by volume of bounding box: {volume}")

def plot(min_val, max_val):
    box_vertices = []
    box_faces = []

    fig = plt.figure()
    graph = fig.add_subplot(111, projection='3d')

    # Define the 8 vertices of the box
    box_vertices = np.array([[min_val[0], min_val[1], min_val[2]],
                              [max_val[0], min_val[1], min_val[2]],
                              [max_val[0], max_val[1], min_val[2]],
                              [min_val[0], max_val[1], min_val[2]],
                              [min_val[0], min_val[1], max_val[2]],
                              [max_val[0], min_val[1], max_val[2]],
                              [max_val[0], max_val[1], max_val[2]],
                              [min_val[0], max_val[1], max_val[2]]])

    # Define the 6 faces of the box
    box_faces = [[box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]],
                 [box_vertices[4], box_vertices[5], box_vertices[6], box_vertices[7]],
                 [box_vertices[0], box_vertices[1], box_vertices[5], box_vertices[4]],
                 [box_vertices[1], box_vertices[2], box_vertices[6], box_vertices[5]],
                 [box_vertices[2], box_vertices[3], box_vertices[7], box_vertices[6]],
                 [box_vertices[3], box_vertices[0], box_vertices[4], box_vertices[7]]]
    
    # Plot the box faces
    graph.add_collection3d(Poly3DCollection(box_faces, facecolors='pink', linewidths=1, edgecolors='w', alpha=.25))
    
    # Plot each prism
    for prism in prisms:
        faces = [[prism[0], prism[1], prism[2], prism[3]],
                 [prism[4], prism[5], prism[6], prism[7]],
                 [prism[0], prism[1], prism[5], prism[4]],
                 [prism[2], prism[3], prism[7], prism[6]],
                 [prism[1], prism[2], prism[6], prism[5]],
                 [prism[4], prism[7], prism[3], prism[0]]]
        graph.add_collection3d(Poly3DCollection(faces, facecolors='brown', linewidths=1, edgecolors='black'))

    # Set labels and display
    graph.set_xlim([min_val[0], max_val[0]])
    graph.set_ylim([min_val[1], max_val[1]])
    graph.set_zlim([min_val[2], max_val[2]])
    graph.set_xlabel('X')
    graph.set_ylabel('Y')
    graph.set_zlabel('Z')
    graph.set_title('Geometric Pancakes')
    plt.show()

if __name__ == "__main__":
    min_val, max_val, prisms, deg_acc = pre_processing()
    results, deg_acc = collecting(min_val, max_val, prisms, deg_acc)
    reducer(results, min_val, max_val, deg_acc)
    plot(min_val, max_val)