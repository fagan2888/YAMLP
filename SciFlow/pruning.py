"""This module generates a pruned set of the data set. The procedure of pruning is as follows:

1. Import the data and have a low level energy associated with each sample.

2. Run a clustering algorithm. The centres of each cluster are what is kept as a point in the pruned data set.

3. Run a furthest first traversal algorithm to keep the samples that are as different from each other as possible.

4. Calculate the energy at a higher level of theory and calculate the energy difference for each point in the pruned
data set. Plot the distribution of the errors and decide what to do with the outliers.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Pruning():
    """
    This class contains functions that help pruning the data that will then be used to fit a neural network.

    It takes as an input the X and the y part of the data. The X part can contain only the coordinates (not the atom labels).
    The y part contains the energies.

    It has been created with the idea that it will be used in a jupyter notebook.

    :X: array of shape (n_samples, dim x n_atoms)
    :y: numpy array of shape (n_samples,)
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.dim = X.shape[1]


    def elbow(self,n_centres):
        """
        This function does the elbow procedure to work out the best number of centres to use. The n_centres parameter
        contains a list with a list of values to try. It makes a plot to let the user decide the best number of centres
        to use.

        :n_centres: list of int
        """

        tot_sum_of_sq = []

        for i in n_centres:
            kmeans = KMeans(n_clusters=i).fit(self.X)
            clusters_idx = kmeans.predict(self.X)  # indices of which cluster each point belongs to
            centres = kmeans.cluster_centers_

            sum_of_squares = 0

            for j, item in enumerate(clusters_idx):
                dist = euclidean_distances(self.X[i].reshape(-1, 1), centres[item].reshape(-1, 1))[0][0]
                sum_of_squares = sum_of_squares + dist ** 2

            tot_sum_of_sq.append(sum_of_squares)

        self.__plot_elbow(n_centres, tot_sum_of_sq)

    def get_X(self):
        return self.X

    def clustering(self, n_clusters):
        """
        This function clusters the data into n_clusters and returns the indexes of the data points in each cluster that
        are closest to the centre.

        :n_clusters: int
        :return: array of int
        """
        if n_clusters < 5000:
            kmeans = KMeans(n_clusters=n_clusters).fit(self.X)
        else:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(self.X)

        clusters_idx = kmeans.predict(self.X)  # indices of which cluster each point belongs to
        self.centres = kmeans.cluster_centers_
        dist_mat = kmeans.transform(self.X)  # (n_samples, n_clusters) matrix of distances of each sample to each centre

        self.idx_clust = np.zeros((n_clusters,))

        for i in range(n_clusters):
            self.idx_clust[i] = np.argmin(dist_mat[:,i])

        self.idx_clust = self.idx_clust.astype(int)

        self.X_cl = np.zeros((len(self.idx_clust), self.X.shape[1]))
        for i, item in enumerate(self.idx_clust):
            self.X_cl[i, :] = self.X[item, :]

        if self.dim == 2:
            self.__plot_centres()

        return self.idx_clust

    def __plot_centres(self):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.X[:,0],self.X[:,1], label="Points", color="yellow")
        ax.scatter(self.centres[:,0],self.centres[:,1], label="Centres", color="black")
        ax.scatter(self.X_cl[:, 0], self.X_cl[:, 1], label="Points to keep", color="red")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        plt.show()

    def __plot_fft(self):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.X[:,0],self.X[:,1], label="Points", color="yellow")
        ax.scatter(self.centres[:, 0], self.centres[:, 1], label="Centres", color="black")
        ax.scatter(self.X_fft[:, 0], self.X_fft[:, 1], label="Points to keep", color="red")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        plt.show()

    def __plot_elbow(self,n_centres,tot_sum_of_sq):

        k_df = pd.DataFrame()
        k_df['n of clusters'] = n_centres
        k_df['sum of squares'] = tot_sum_of_sq
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.pointplot(x="n of clusters", y="sum of squares", data=k_df)
        ax.set_ylabel('Sum of distance squares')
        ax.set_xlabel('Number of clusters')
        plt.show()

    def fft_idx(self, n_points, save=False):
        """
        This function goes through all the points in the data set and returns the **indices** of the samples to put into
        the training set. n_points is the number of points that should be put into the training set. There is the option
        of printing to a file the indexes of the samples to keep.

        :n_points: int (smaller than n_samples)
        :save: bool
        :return: list of int
        """

        # Creating the matrix of the distances
        dist_mat = euclidean_distances(self.X_cl, self.X_cl)

        n_samples = self.X_cl.shape[0]

        self.idx_fft = np.zeros((n_points, )).astype(int)

        # Adding a first random sample to the set
        idx = np.int32(np.random.uniform(n_samples))
        self.idx_fft[0] = idx

        for i in range(1, n_points):
            dist_list = []
            for index in self.idx_fft:
                dist_list.append(dist_mat[index, :])
            dist_set = np.amin(dist_list, axis=0)
            dist_idx = np.int32(np.argmax(dist_set))
            self.idx_fft[i] = dist_idx

        self.X_fft = np.zeros((len(self.idx_fft), self.dim))
        for i, item in enumerate(self.idx_fft):
            self.X_fft[i, :] = self.X_cl[item, :]


        if save == True:
            np.save("idx_fft.npy",self.idx_fft)

        if self.dim == 2:
            self.__plot_fft()

        return self.idx_fft

if __name__ == "__main__":

    from sklearn.datasets.samples_generator import make_blobs

    centers = [[18, 18], [-18, -18], [18, -18]]
    n_clusters = len(centers)
    X, y = make_blobs(n_samples=8000, centers=centers, cluster_std=5)

    pr = Pruning(X, y)
    # pr.elbow(range(1,30))
    n_clusters = 6000
    idx_clust = pr.clustering(n_clusters=n_clusters)
    idx_fft = pr.fft_idx(50)



