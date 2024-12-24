
### TODO 1: Importing the necessary libraries - numpy, matplotlib and time
import numpy as np
import matplotlib.pyplot as plt
import time
import random


### TODO 2
### Load data from data_path
### Check the input file spice_locations.txt to understand the Data Format
### Return : np array of size Nx2
def load_data(data_path):
    coordinates=[]
    with open(data_path, 'r') as file:
        for line in file:
            x,y= line.strip().split(',')
            coordinates.append((float(x),float(y)))
    return np.array(coordinates)


### TODO 3.1
### If init_centers is None, initialize the centers by selecting K data points at random without replacement
### Else, use the centers provided in init_centers
### Return : np array of size Kx2
def initialise_centers(coords, K, init_centers=None):
    if init_centers!=None:
        return init_centers
    else:
        centres=np.array(random.sample(coords.tolist(), K))
        return centres
 

### TODO 3.2
### Initialize the labels to all ones to size (N,) where N is the number of data points
### Return : np array of size N
def initialise_labels(data):
    pass

### TODO 4.1 : E step
### For Each data point, find the distance to each center
### Return : np array of size NxK
def calculate_distances(coords, centers):
    distances=np.zeros((len(coords),len(centers)), dtype=float)
    for i, (x, y) in enumerate(coords):
        for j, (a, b) in enumerate(centers):
            distances[i,j]=(x-a)**2+(y-b)**2
    return distances

### TODO 4.2 : E step
### For Each data point, assign the label of the nearest center
### Return : np array of size N
def update_labels(distances):
    labels=np.zeros(distances.shape[0],dtype=np.int8)
    for i in range (0,len(labels)):
        labels[i]= np.argmin(distances[i])
    return (labels)


### TODO 5 : M step
### Update the centers to the mean of the data points assigned to it
### Return : np array of size Kx2
def update_centers(coords, labels, K):
    sum_arr=np.zeros((K,2))
    label_count=np.zeros(K)
    for i in range (0, len(labels)):
        sum_arr[labels[i]]+=coords[i]
        label_count[labels[i]]+=1
    for i in range (0, K):
        sum_arr[i]/=label_count[i]
    return sum_arr

### TODO 6 : Check convergence
### Check if the labels have changed from the previous iteration
### Return : True / False
def check_termination(centres, old_centres):
    center_shift = np.linalg.norm(centres-old_centres, axis=1)
    if np.max(center_shift) < 1e-4:
        return True

### simulate the algorithm in the following function. run.py will call this
### function with given inputs.
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    start_time=time.time()
    max_iterations=300
    coords=load_data(data_path)
    centres=initialise_centers(coords,K)
    old_centres=initialise_centers(coords,K)
    i=int(0)
    while ((not check_termination(centres,old_centres)) and i<max_iterations):
        i+=1
        old_centres=centres
        distances=calculate_distances(coords,centres)
        labels= update_labels(distances)
        centres=update_centers(coords, labels, K)
    distances=calculate_distances(coords,centres)
    labels= update_labels(distances)
    end_time=time.time()
    time_taken=end_time-start_time
    return (centres, labels, time_taken)


    
    
### to visualise the final data points and centers.
def visualise(data_path, labels, centers):
    data = load_data(data_path)
    # Scatter plot of the data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    return plt
