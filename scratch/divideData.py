import numpy as np

def divideData(X, Y, percentages):
    np.random.shuffle(X)
    np.random.shuffle(Y)

    n_samples = Y.shape[0]
    setSizes = n_samples * percentages
    setSizes = setSizes.astype(int)


    # This checks that the sum of the number of samples in each set matches the total number of samples
    # if the number of samples is larger than the total number of samples, some samples are removed from the training
    # dataset. In the opposite case, they are added to the training dataset.
    if np.sum(setSizes) > n_samples:
        setSizes[0] -= (np.sum(setSizes) - n_samples)
    elif np.sum(setSizes) < n_samples:
        setSizes[0] += (n_samples - np.sum(setSizes))

    # Taking the first part of X and Y for the training set
    X_train = X[0:setSizes[0], :]
    Y_train = Y[0:setSizes[0], :]

    X_crossVal = X[setSizes[0]:(setSizes[1]+setSizes[0]), :]
    Y_crossVal = Y[setSizes[0]:(setSizes[1]+setSizes[0]), :]

    X_val = X[(setSizes[1]+setSizes[0]):np.sum(setSizes), :]
    Y_val = Y[(setSizes[1]+setSizes[0]):np.sum(setSizes), :]

    splitX = [X_train, X_crossVal, X_val]
    splitY = [Y_train, Y_crossVal, Y_val]

    return splitX, splitY
