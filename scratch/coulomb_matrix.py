""" This script turns the cartesian representation of a molecule into a Coulomb matrix"""


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def coulombMatrix(fileX):

    # This is a dictionary that contains the nuclear charges of the atoms involved

    Z = {
        'C': 6.0,
        'H': 1.0,
        'N': 7.0}

    # Variables needed for the loop logic
    isFirstTime = True
    n_samples = 0       # number of training samples

    # Setting up plotting of the matrix
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # This reads the first line to establish the number of atoms present and then counts the number of samples
    for line in fileX:
        if(isFirstTime):
            listLine = line.split(",")
            numAtoms = int(len(listLine) / 4)
            isFirstTime = False

        n_samples += 1

    fileX.seek(0)       # Taking the cursor back to the beginning of the file

    ### This calculates the Coulomb matrices for each sample
    cMatrix = np.zeros((numAtoms, numAtoms))            # This is the empty coulomb matrix
    sampleCMatrix = np.zeros((n_samples, numAtoms**2))  # This array contains the flattened Coulomb matrix for each sample
    lineCount = 0                                       # This counts at which line of the file the cursor is

    for line in fileX:

        listLine = line.split(",")
        labels = ""         # This is a string containing all the atom labels
        coord = []          # This is a list of tuples with the coordinates of all the atoms in a configuration

        # This gathers the atom labels and the coordinates for one sample
        for i in range(0,len(listLine)-1,4):
            labels += listLine[i]
            coord.append( np.array([float(listLine[i+1]), float(listLine[i+2]), float(listLine[i+3])]) )

        # Diagonal elements
        for i in range(numAtoms):
            cMatrix[i, i] = 0.5 * Z[labels[i]]**2.4

        # Off-diagonal elements
        for i in range(numAtoms-1):
            for j in range(i+1, numAtoms):
                distanceVec = coord[i] - coord[j]
                distance = np.sqrt(np.dot(distanceVec, distanceVec))
                cMatrix[i, j] = Z[labels[i]]*Z[labels[j]] / distance
                cMatrix[j, i] = cMatrix[i, j]

        # Sorting the Coulomb matrix rows in descending order of the norm of each row.
        rowNorms = np.zeros(numAtoms)
        for i in range(numAtoms):
            rowNorms[i] = LA.norm(cMatrix[i,:])

        permutations = np.argsort(rowNorms)
        permutations = permutations[::-1]

        cMatrix = cMatrix[permutations, :]

        # Populating the 2D array with the coulomb matrix for each sample (n_samples x n_atoms^2)
        sampleCMatrix[lineCount, :] = cMatrix.flatten()
        lineCount += 1

    plt.matshow(sampleCMatrix, fignum=False, )
    plt.show()

    return sampleCMatrix

