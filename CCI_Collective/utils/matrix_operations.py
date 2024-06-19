import numpy as np

def collapse_matrix(matrix):
    return np.array(matrix.sum(axis=0)).squeeze()