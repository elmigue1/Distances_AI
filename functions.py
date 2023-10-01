import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import random
from collections import Counter
import os

def lp_distance(x,y, p = 4):
  if p == np.inf:
      return max(abs(x[i] - y[i]) for i in range(len(x)))
  else:
      return (sum((abs(x[i] - y[i])**p) for i in range(len(x))))**(1/p)

def cosine_distance(x, y, p=4):
    return 1 - np.dot(x, y) / (np.linalg.norm(x,ord = p) * np.linalg.norm(y,ord = p))

def maha2_distance(X,point):
    u = np.mean(X, axis=0)
    distance_matrix = np.zeros(X.shape[0])
    cov = np.linalg.inv(np.cov(X, rowvar=False))
    for i in range(X.shape[0]):
      distance_matrix[i] = np.sqrt((X[i]-point).T.dot(cov).dot((X[i]-point)))
    return distance_matrix

def maha_distance(X,x,y):
    u = np.mean(X, axis=0)
    distance_matrix = np.zeros(X.shape[0])
    cov = np.linalg.inv(np.cov(X, rowvar=False))
    return np.sqrt((x-y).T.dot(cov).dot((x-y)))


def distancemat(x, distance=maha_distance, groups=4, kn=10):
    dims = x.shape[1]
    N = x.shape[0]
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if distance == maha_distance:
                matrix[i, j] = distance(x, x[i], x[j])
            else:
                matrix[i, j] = distance(x[i], x[j])
    matrix = matrix + matrix.T
    return matrix

def bygroup(matrix,g):
  N = len(matrix)
  groups = np.zeros(N, dtype = int)
  points = set(range(N))
  reference = random.choice(tuple(points))
  points.remove(reference)
  l = np.min(matrix)
  u = np.max(matrix)
  span = u-l
  tresh = span/g
  for i in points:
    dist = matrix[reference,i]
    groups[i] = (dist//tresh)
  return groups

def kn(matrix, k):
    N = len(matrix)
    groups = np.zeros(N, dtype = int)  # Initialize groups with zeros (unassigned)
    group_number = 0  # Start with group 1
    unassigned_points = set(range(N))
    while unassigned_points:
        # Pick a random unassigned point
        current_point = random.choice(tuple(unassigned_points))
        unassigned_points.remove(current_point)

        # Get distances from the current point to all other points
        distances = matrix[current_point]

        # Get the indices of the k closest unassigned points
        closest_points = sorted(unassigned_points, key=lambda x: distances[x])[:k]

        # Assign the current point and its k closest unassigned points to the current group
        groups[current_point] = group_number
        groups[closest_points] = group_number

        # Remove these points from the unassigned points set
        unassigned_points -= set(closest_points)

        # Increment the group number for the next group
        group_number += 1

    return groups

def final_groups(matrix, fun , k, n_iterations):
    N = len(matrix)
    group_occurrences = [Counter() for _ in range(N)]

    for _ in range(n_iterations):
        groups = fun(matrix, k)
        for point, group in enumerate(groups):
            group_occurrences[point][group] += 1

    final_groups = np.zeros(N, dtype=int)
    for point, counter in enumerate(group_occurrences):
        final_groups[point] = counter.most_common(1)[0][0]

    return final_groups

