import importData
import numpy as np
from numpy import linalg as LA


class CoulombMatrix():
    """
    This is the class that creates Coulomb matrix descriptors.

    __init__: initialises the following quantities:
        Z = The dictionary with nuclear charges
        n_atoms = the number of atoms in the system
        n_samples = the number of samples contained in the data
        coulMatrix = the empty (n_samples, n_atoms^2) np coulomb matrix

    generate: it generates values in the coulomb matrix
    """

    def __init__(self, inputFile = None, matrixX = None):

        # Checking that the correct arguments have been given:
        if inputFile == None and matrixX == None:
            print "Error: you need to give an imput in order to generate a Coulomb matrix. \n"
            print "Either specify an XML file with the data or pass a list with the atom labels and coordinates for each atom. \n"
            quit()
        elif inputFile != None and matrixX != None:
            print "Error: you have given both a XML file and a matrix for input. Give one or the other."
            quit()

        # Depending on whether inputFile or matrixX have been given, the list rawX is initialised differently:
        if inputFile != None:
            importData.XMLtoCSV(inputFile)
            self.rawX = importData.loadX("X.csv")
        else:
            self.rawX = matrixX

        self.Z = {
                    'C': 6.0,
                    'H': 1.0,
                    'N': 7.0
                 }

        self.n_atoms = len(self.rawX[0])/4
        self.n_samples = len(self.rawX)

        self.coulMatrix = np.zeros((self.n_samples, self.n_atoms**2))

    def generate(self):
        """
        This function generates the Coulomb Matrix descriptor as a numpy array of size (n_samples, n_atoms^2)
        :return: None
        """

        # This is a coulomb matrix for one particular sample in the dataset
        indivCM = np.zeros((self.n_atoms, self.n_atoms))
        sampleCount = 0

        for item in self.rawX:

            labels = ""  # This is a string containing all the atom labels
            coord = []  # This is a list of tuples with the coordinates of all the atoms in a configuration

            # This gathers the atom labels and the coordinates for one sample
            for i in range(0, len(item), 4):
                labels += item[i]
                coord.append(np.array([float(item[i + 1]), float(item[i + 2]), float(item[i + 3])]))

            # Diagonal elements
            for i in range(self.n_atoms):
                indivCM[i, i] = 0.5 * self.Z[labels[i]] ** 2.4

            # Off-diagonal elements
            for i in range(self.n_atoms - 1):
                for j in range(i + 1, self.n_atoms):
                    distanceVec = coord[i] - coord[j]
                    distance = np.sqrt(np.dot(distanceVec, distanceVec))
                    indivCM[i, j] = self.Z[labels[i]] * self.Z[labels[j]] / distance
                    indivCM[j, i] = indivCM[i, j]

            # Sorting the Coulomb matrix rows in descending order of the norm of each row.
            rowNorms = np.zeros(self.n_atoms)
            for i in range(self.n_atoms):
                rowNorms[i] = LA.norm(indivCM[i, :])

            permutations = np.argsort(rowNorms)
            permutations = permutations[::-1]

            indivCM = indivCM[permutations, :]

            # The coulomb matrix for each sample is flattened and added to the total coulomb matrix
            self.coulMatrix[sampleCount, :] = indivCM.flatten()
            sampleCount += 1

        print "Generated the Coulomb Matrix. \n"

    def normalise_1(self):
        """
        This function normalises the Coulomb matrix to make learning more efficient.
        It uses the maximum and the minimum element in the features.
        :return: None
        """
        # Each element contains the min/max value of each 'feature' present in the Coulomb matrix
        p_min = np.amin(self.coulMatrix, axis=0)
        p_max = np.amax(self.coulMatrix, axis=0)

        p_diff = p_max - p_min

        for i in range(len(p_diff)):
            if p_diff[i] == 0:
                p_diff[i] += 1e-7

        numerator = 2 * (self.coulMatrix - p_min)

        self.coulMatrix = numerator / p_diff -1

        return None


    def normalise_2(self):
        """
        This function normalises the Coulomb matrix to make learning more efficient.
        It uses the std deviation and the mean
        :return: None
        """
        # Each element contains the min/max value of each 'feature' present in the Coulomb matrix
        p_mean = np.mean(self.coulMatrix, axis=0)
        p_std = np.std(self.coulMatrix, axis=0)

        for i in range(len(p_std)):
            if p_std[i] == 0:
                p_std[i] += 1e-7

        self.coulMatrix = 2 * (self.coulMatrix - p_mean) / p_std

        return None



if __name__ == "__main__":
    importData.XMLtoCSV("/Users/walfits/Repositories/tensorflow/AMP/input1.xml")
    X = importData.loadX("X.csv")
    cm = CoulombMatrix(matrixX=X)
    cm.generate()

    # Generating a plot of matrix
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    anExample = np.reshape(cm.coulMatrix[3,:], (7,7))
    # plt.matshow(cm.coulMatrix, fignum=False)
    plt.matshow(anExample, fignum=False)
    plt.show()