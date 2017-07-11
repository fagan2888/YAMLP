import ImportData
import numpy as np
import CoulombMatrix
import cProfile, pstats, StringIO
import time
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx

def fft_idx(X, k):

    # Creating the matrix of the distances
    dist_mat_glob = np.zeros(shape=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            distvec = X[j, :] - X[i, :]
            dist_mat_glob[i, j] = np.dot(distvec, distvec)
            dist_mat_glob[j, i] = np.dot(distvec, distvec)

    n_samples = X.shape[0]
    train_set = []

    idx = np.int32(np.random.uniform(n_samples))
    train_set.append(idx)

    for i in range(1, k):
        dist_list = []
        for index in train_set:
            dist_list.append(dist_mat_glob[index, :])
        dist_set = np.amin(dist_list, axis=0)
        dist_idx = np.argmax(dist_set)
        train_set.append(dist_idx)

    return train_set

def fft_split_np(X, y, size=0.8):
    """
    This function splits the data set according to the farthest-first traversal algorithm.
    :param X: the dataset (n_samples, n_features)
    :param y: the labels to the dataset (n_samples, )
    :param size: The percentage of datapoints to put into the training set
    :return: X_train (n_samples*size, n_features), X_test (n_samples*(1-size), n_features), y_train, y_test
    """

    pr = cProfile.Profile()
    pr.enable()

    # Reshaping y to be of correct dimensions for tensorflow
    y = np.reshape(y,(len(y), 1))

    # Attaching some indexes to the dataset
    idx = range(0,X.shape[0])
    idx = np.reshape(idx, (len(idx), 1))

    global dist_mat_glob
    dist_mat_glob = np.zeros(shape=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            distvec = X[j, :] - X[i, :]
            dist_mat_glob[i, j] = np.dot(distvec, distvec)
            dist_mat_glob[j, i] = np.dot(distvec, distvec)
    #
    # dist_graph = nx.from_numpy_matrix(dist_mat_glob)
    # print dist_graph.number_of_nodes()

    # Concatenating dataset to make splitting easier
    Xy = np.concatenate((X, idx, y), axis=1)

    # Useful numbers
    n_samples = X.shape[0]
    n_feat = X.shape[1]
    n_train = int(n_samples * size)
    n_test = int(n_samples * (1 - size))

    # Making empty arrays that will contain the test set and train set
    Xy_train = np.zeros((int(n_samples*size), n_feat+2))
    # Xy_test = np.zeros((int(n_samples * (1-size)), n_feat + 1))

    # Finding the equilibrium structure by taking the one with lowest energy. Then putting it in train set
    idx_min = np.argmin(Xy[:,-1])
    Xy_train[0,:] = Xy[idx_min, :]
    Xy = np.delete(Xy, (idx_min), axis=0)

    # This loop goes over all the spaces to be filled in the training set
    for i in range(1, n_train):
        dist_to_train = np.zeros(Xy.shape[0])   # Array with the distance of each Xy sample to the set Xy_train

        # This loop calculates the distances that each sample in the data set has to the training set
        for j in range(Xy.shape[0]):
            all_dist = np.zeros(Xy_train.shape[0])      # Distance of a particular Xy sample to each Xy_train samples
            for k in range(Xy_train.shape[0]):
                idx1 = int(Xy[j,-2])
                idx2 = int(Xy_train[k,-2])
                all_dist[k] = dist_mat_glob[idx1, idx2]
            # The distance of a sample in Xy to the set Xy_train is the minimum of the all_dist array
            dist_to_train[j] = np.amin(all_dist)
        idx_max = np.argmax(dist_to_train)
        Xy_train[i, :] = Xy[idx_max, :]
        Xy = np.delete(Xy, (idx_max), axis=0)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

# Reference functions that were constructed before it was realised that calling functions is so slow
def point_point_dist(x1, x2):
    distvec = x2 - x1
    dist = np.dot(distvec, distvec)
    return dist

def point_point_dist_quick(idx1, idx2):
    return max(dist_mat_glob[idx1, idx2], dist_mat_glob[idx2, idx1])

def point_set_dist_quick(x, C):
    all_dist = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        all_dist[i] = point_point_dist_quick(x[-1], C[i, -1])
    dist = np.amin(all_dist)
    return dist

def point_set_dist(x, C):
    all_dist = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        all_dist[i] = point_point_dist(x, C[i])
    dist = np.amin(all_dist)
    return dist

def dist_matrix(X):
    """This should take about a minute for 35000 data points"""
    dist_mat = np.zeros(shape=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0]-1):
        for j in range(i+1,X.shape[0]):
            # dist_mat[i, j] = point_point_dist(X[i, :], X[j, :])
            distvec = X[j, :]-X[i, :]
            dist_mat[i, j] = np.dot(distvec, distvec)

    return dist_mat






if __name__ == "__main__":
    X, y, Q = ImportData.loadPd_q("/Users/walfits/Repositories/trainingNN/dataSets/PBE_B3LYP/pbe_b3lyp_partQ_rel.csv")
    descript = CoulombMatrix.CoulombMatrix(X)
    # X_coul, y_coul = descript.generatePRCM(y_data=y, numRep=2)
    X_coul = descript.generateTrimmedCM()

    pr = cProfile.Profile()
    pr.enable()

    train_idx = fft_idx(X_coul[:1000, :], 800)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


    # x = range(100, 1100, 100)
    # y = []
    #
    # for i in range(100, 1100, 100):
    #     X_set = X_coul[:i, :]
    #     # Starting the timer
    #     startTime = time.time()
    #     fft_idx(X_coul[:i,:], int(i*0.8))
    #     # Ending the timer
    #     endTime = time.time()
    #     finalTime = endTime - startTime
    #     y.append(finalTime)
    #
    # fig2, ax2 = plt.subplots(figsize=(6, 6))
    # ax2.scatter(x, y)
    # ax2.set_xlabel('Data set size')
    # ax2.set_ylabel('Time to split (s)')
    # ax2.legend()
    # plt.show()
    #
    # print x
    # print y