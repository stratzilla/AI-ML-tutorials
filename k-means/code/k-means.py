#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import random
import math

INF = float('inf')
SEED = 5

random.seed(SEED)

class Point:
    def __init__(self, x, y):
        self.x, self.y, self.c = x, y, None
    
    def set_coordinate(self, x, y):
        self.x, self.y = x, y
    
    def get_coordinate(self):
        return self.x, self.y
    
    def set_cluster(self, c):
        self.c = c
    
    def get_cluster(self):
        return self.c
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __neq__(self, other):
        return not self.__eq__(other)

def k_means(epochs):
    points = load_data('./data/s1.txt')
    clusters = initialize_centroids(points, 15)
    for _ in range(epochs):
        find_clusters(points, clusters)
        move_centroids(points, clusters)
    print(f"DI = {dunn_index(points, clusters)}")
    plot_data(points, clusters)

def dunn_index(points, clusters):
    min_cluster_dist = INF
    for c_i in clusters:
        for c_j in clusters:
            if c_i is c_j:
                continue
            min_cluster_dist = min(min_cluster_dist, distance(c_i, c_j))
    max_cluster_diam = 0
    for c in clusters:
        for p_i in points:
            if p_i.get_cluster() is not c:
                continue
            for p_j in points:
                if p_j.get_cluster() is not c:
                    continue
                if p_i is p_j:
                    continue
                max_cluster_diam = max(max_cluster_diam, distance(p_i, p_j))
    return round(min_cluster_dist/max_cluster_diam, 4)

def distance(p, q):
    (p_x, p_y), (q_x, q_y) = p.get_coordinate(), q.get_coordinate()
    return math.sqrt((q_x - p_x)**2 + (q_y - p_y)**2)

def find_clusters(points, clusters):
    for p in points:
        best_distance, best_cluster = INF, None
        for c in clusters:
            d = distance(p, c)
            if d < best_distance:
                best_distance, best_cluster = d, c
        p.set_cluster(best_cluster)

def initialize_centroids(points, k):
    clusters = []
    clusters.append(random.choice(points))
    for _ in range(k-1):
        distances = []
        for p in points:
            d = INF
            for c in clusters:
                d = min(d, distance(p, c))
            distances.append(d)
        clusters.append(points[distances.index(max(distances))])
    return clusters

def move_centroids(points, clusters):
    for c in clusters:
        mean_x, mean_y, count = 0, 0, 0
        for p in points:
            if p.get_cluster() is not c:
                continue
            x, y = p.get_coordinate()
            mean_x += x
            mean_y += y
            count += 1
        new_x, new_y = mean_x/count, mean_y/count
        c.set_coordinate(new_x, new_y)

def load_data(file):
    df = pd.read_csv(file, names=['x', 'y'], delimiter=r'\s+')
    points = []
    for _, d in df.iterrows():
        points.append(Point(d['x'], d['y']))
    return points
	
def plot_data(points, clusters):
    colors = ['red', 'lime', 'blue', 'yellow', 'orange', 'deeppink', \
        'olivedrab', 'aqua', 'thistle', 'mediumvioletred', 'plum', \
        'burlywood', 'maroon', 'mediumspringgreen', 'dodgerblue', \
        'rebeccapurple', 'lightcoral', 'darkslategrey', 'firebrick', 'bisque', \
        'darkseagreen', 'fuchsia', 'turquoise', 'steelblue', 'chocolate']
    plt.xticks([]); plt.yticks([])
    plt.margins(0.05, 0.05)
    for c, i in zip(clusters, range(len(clusters))):
        point_x, point_y = [], []
        for p in points:
            if p.get_cluster() is not c:
                continue
            x, y = p.get_coordinate()
            point_x.append(x); point_y.append(y)
        plt.scatter(point_x, point_y, c=colors[i], s=2, marker='o', lw='1')
    cluster_x, cluster_y = [], []
    for c, i in zip(clusters, range(len(clusters))):
        x, y = c.get_coordinate()
        plt.scatter(x, y, c=colors[i], s=100, marker='X', lw='1', ec='k')
    plt.show()
    plt.clf()

if __name__ == '__main__':
	k_means(15)