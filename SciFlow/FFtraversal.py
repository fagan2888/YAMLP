"""
This module contains functions that split the data set using a furthest first traversal procedure.
"""


import ImportData
import numpy as np
import CoulombMatrix
import cProfile, pstats, StringIO
import time
from datetime import datetime
import matplotlib.pyplot as plt

def fft_idx(X, k):
    """
    This function goes through all the points in the data set and returns the indices of the samples to put into the
    training set. X is the data set to split while k is the number of points that should be put into the training set.

    :X: numpy array of shape (n_samples, n_features)
    :k: int (smaller than n_samples)
    :return: list of int
    """

    # Creating the matrix of the distances
    dist_mat_glob = np.zeros(shape=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            distvec = X[j, :] - X[i, :]
            dist_mat_glob[i, j] = np.dot(distvec, distvec)
            dist_mat_glob[j, i] = np.dot(distvec, distvec)

    # print "Generated " + str(X.shape[0]) + " by " + str(X.shape[0]) + " distance matrix."

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

    # np.save("train_idx.npy",train_set)
    return train_set




if __name__ == "__main__":
    X, y, Q = ImportData.loadPd_q("/Users/walfits/Repositories/trainingNN/dataSets/PBE_B3LYP/pbe_b3lyp_partQ_rel.csv")
    descript = CoulombMatrix.CoulombMatrix(X)
    # X_coul, y_coul = descript.generatePRCM(y_data=y, numRep=2)
    X_coul = descript.generateTrimmedCM()
    print X_coul.shape

    pr = cProfile.Profile()
    pr.enable()

    train_idx = fft_idx(X_coul[:5000, :], 4000)

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