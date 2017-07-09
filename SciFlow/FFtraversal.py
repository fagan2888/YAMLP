import tensorflow as tf
import ImportData
import numpy as np
import PartialCharge

def fft_split_np(X, y, size=0.8):
    """
    This function splits the data set according to the farthest-first traversal algorithm.
    :param X: the dataset (n_samples, n_features)
    :param y: the labels to the dataset (n_samples, )
    :param size: The percentage of datapoints to put into the training set
    :return: X_train (n_samples*size, n_features), X_test (n_samples*(1-size), n_features), y_train, y_test
    """

    # Reshaping y to be of correct dimensions for tensorflow
    y = np.reshape(y,(len(y), 1))

    # Concatenating dataset to make splitting easier
    Xy = np.concatenate((X, y), axis=1)

    # Useful numbers
    n_samples = X.shape[0]
    n_feat = X.shape[1]
    n_train = int(n_samples * size)
    n_test = int(n_samples * (1 - size))

    # Making empty arrays that will ocntain the test set and train set
    Xy_train = np.zeros((int(n_samples*size), n_feat+1))
    Xy_est = np.zeros((int(n_samples * (1-size)), n_feat + 1))

    # Finding the equilibrium structure by taking the one with lowest energy. Then putting it in train set
    idx_min = np.argmin(Xy[:,-1])
    Xy_train[0,:] = Xy[idx_min, :]
    Xy = np.delete(Xy, (idx_min), axis=0)

    # Finding the point in Xy that is the furthest away from the point in Xy_train, then add it to Xy_train

    # Follow procedure from here http://web.cse.ohio-state.edu/mlss09/mlss09_talks/8.june-MON/clustering-dasgupta.pdf












if __name__ == "__main__":
    X, y, Q = ImportData.loadPd_q("/Users/walfits/Repositories/trainingNN/dataSets/PBE_B3LYP/pbe_b3lyp_partQ_rel.csv")
    PCCM = PartialCharge.PartialCharges(X, y, Q)
    X_coul, y_coul = PCCM.generatePCCM(numRep=2)
    fft_split_np(X_coul, y_coul)