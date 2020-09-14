#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

def read_data(path):
    """
    Read the input file and store it in data_set.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        path: path to the dataset

    Returns:
        data_set: n_samples x n_features
            A list of data points, each data point is itself a list of features.
    """
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():             
            line = line.strip()                

            x = line.split(",")
            temp = []
            for item in x:
                temp.append(float(item))
            
            data.append(temp)
            # print(data)
    # print(data)
    return np.asarray(data)

def pca(data_set, n_components):
    """
    Perform principle component analysis and dimentinality reduction.
    
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        n_components: int
            The number of components to keep. If n_components is None, all components should be kept.

    Returns:
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.
    """
    # Step 1: zero normalization
    normed = (data_set - data_set.mean(axis=0))
    # Step 2: Calculate the covariance matrix
    cov_mat = np.cov(np.transpose(normed))
    # Step 3: Find the eigenvectors and eigenvalues of the covariance matrix
    w, v = np.linalg.eigh(cov_mat)
    # Step 4: The eigenvectors (sorted by the descending order of eigenvalues) are the Principal Components
    indice_need = np.argsort(w)[::-1] # get indice of first n elements
    v_sorted = v[:, indice_need]
    
    return (v_sorted[:, : n_components])
def dim_reduction(data_set, components):
    """
    perform dimensionality reduction (change of basis) using the components provided.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.

    Returns:
        transformed: n_samples x n_components
            Return the transformed values.
    """
    transformed = np.matmul(data_set, components)
    return transformed

# You may put code here to test your program. They will not be run during grading.
if __name__ == '__main__':
    data_set = read_data("pizza.txt")
    components = pca(data_set, 2)
    a = dim_reduction(data_set, components)
    fig, ax = plt.subplots()
    ax.plot(a[:,0], a[:,1], '.')
    plt.tight_layout()
    fig.savefig("result.png")
