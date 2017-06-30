import numpy as np
from numpy import linalg as LA

class CoulombMatrix():
    """
    This is the class that creates Coulomb matrix and then has functions to create the Eigen spectrum, the sorted coulomb
    matrix and the randomly sorted coulomb matrix descriptors.

    __init__: initialises the following quantities:
        rawX: list of lists, where each of the inner lists represents a configuration of a sample.
            An example is shown below:
            [ [ 'C', 0.1, 0.3, 0.5, 'H', 0.0, 0.5 1.0, 'H', 0.0, -0.5, -1.0, ....], [...], ... ]
            The labels of the atoms are strings. The coordinates are floats.
        Z: A dictionary of nuclear charges. The keys are the atoms labels.
        n_atoms: number of atoms in the system - int
        n_samples: number of samples collected
        coulMatrix: the flattened coulomb matrix for each sample. - numpy array of size (n_samples, n_atoms^2)


    __generateCM: generates coulMatrix
    generateES: generates the eigen spectrum descriptor from coulMatrix
    generateSCM: generates the sorted coulomb matrix descriptor from coulMatrix
    generateRSCM: generates the randomly sorted coulomb matrix descriptor from coulMatrix
    getCM: returns coulMatrix
    """

    def __init__(self, matrixX = None):
        """
        :param matrixX: list of lists of atom labels and their coordinates.
        """

        self.rawX = matrixX
        self.Z = {
                    'C': 6.0,
                    'H': 1.0,
                    'N': 7.0
                 }

        self.n_atoms = len(self.rawX[0])/4
        self.n_samples = len(self.rawX)

        self.coulMatrix = np.zeros((self.n_samples, self.n_atoms**2))
        self.__generateCM()

        print "Initialised the Coulomb matrix. \n"

    def getCM(self):
        return self.coulMatrix

    def __generateCM(self):
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

            # The coulomb matrix for each sample is flattened and added to the total coulomb matrix
            self.coulMatrix[sampleCount, :] = indivCM.flatten()
            sampleCount += 1

    def generateES(self):
        """
        This function turns the Coulomb matrix into its eigenspectrum.
        :return: numpy array of size (n_samples, n_atoms)
        """

        self.coulES = np.zeros((self.n_samples, self.n_atoms))

        for i in range(self.n_samples):
            tempCM = np.reshape(self.coulMatrix[i, :], (self.n_atoms, self.n_atoms))
            tempES, tempDiag = LA.eig(tempCM)
            self.coulES[i,:] = tempES

        return self.coulES

    def generateSCM(self):
        """
        This function calculates the sorted coulomb matrix starting from the original coulomb matrix.
        :return: the sorted coulomb matrix - numpy array of size (N_samples, n_atoms^2)
        """

        coulS = np.zeros((self.n_samples, int(self.n_atoms * (self.n_atoms+1) * 0.5)))

        for i in range(self.n_samples):
            tempCM = np.reshape(self.coulMatrix[i, :], (self.n_atoms, self.n_atoms))

            # Sorting the Coulomb matrix rows in descending order of the norm of each row.
            rowNorms = np.zeros(self.n_atoms)
            for j in range(self.n_atoms):
                rowNorms[j] = LA.norm(tempCM[j, :])

            permutations = np.argsort(rowNorms)
            permutations = permutations[::-1]

            tempCM = tempCM[permutations, :]
            tempCM = tempCM[:, permutations]
            coulS[i, :] = self.trimAndFlat(tempCM)

        return coulS

    def generateRSCM(self, y_data, numRep=5):
        """
        This function creates the Randomy sorted Coulomb matrix starting from the Coulomb matrix and it transforms the
        y part of the data so that there are numRep copies of each energy.
        :y_data: a numpy array of energy values of size (N_samples,)
        :param numRep: number of randomly sorted matrices to be generated per sample - int
        :return: the randomly sorted CM - numpy array of size (N_samples*numRep, n_atoms^2),
                y_bigdata: a numpy array of energy values of size (N_samples*numRep,)
        """

        # Checking reasonable numRep value
        if(isinstance(numRep, int) == False):
            print "Error: you gave a non-integer value for the number of RSCM that you want to generate."
            return None
        elif(numRep < 1):
            print "Error: you cannot generate less than 1 RSCM per sample. Enter an integer value > 1."

        counter = 0
        coulRS = np.zeros((self.n_samples*numRep, int(self.n_atoms * (self.n_atoms+1) * 0.5)))
        y_bigdata = np.zeros((self.n_samples*numRep,))

        for i in range(self.n_samples):
            tempCM = np.reshape(self.coulMatrix[i, :], (self.n_atoms, self.n_atoms))

            # Calculating the norm vector for the coulomb matrix
            rowNorms = np.zeros(self.n_atoms)

            for j in range(self.n_atoms):
                rowNorms[j] = LA.norm(tempCM[j, :])

            for k in range(numRep):
                # Generating random vectors and adding to the norm vector
                randVec = np.random.normal(loc=0.0, scale=np.std(rowNorms), size=self.n_atoms)
                rowNormRan = rowNorms + randVec
                # Sorting the new random norm vector
                permutations = np.argsort(rowNormRan)
                permutations = permutations[::-1]
                # Sorting accordingly the Coulomb matrix
                tempRandCM = tempCM[permutations, :]
                tempRandCM = tempRandCM[:,permutations]
                # Adding flattened and trimmed randomly sorted Coulomb matrix to the final descriptor matrix
                coulRS[counter, :] = self.trimAndFlat(tempRandCM)
                counter = counter + 1

            # Copying multiple values of the energies
            y_bigdata[numRep*i:numRep*i+numRep] = y_data[i]

        return coulRS, y_bigdata

    def generateTrimmedCM(self):
        """
        This function returns the trimmed version of the Coulomb matrix.
        :return: the trimmed coulomb matrix. NP array of shape (self.n_samples, int(self.n_atoms * (self.n_atoms+1) * 0.5))
        """
        self.trimCM = np.zeros((self.n_samples, int(self.n_atoms * (self.n_atoms+1) * 0.5)))

        for i in range(self.n_samples):
            tempCM = np.reshape(self.coulMatrix[i,:], (self.n_atoms, self.n_atoms))
            self.trimCM[i,:] = self.trimAndFlat(tempCM)

        return self.trimCM

    def trimAndFlat(self, X):
        """
        This function takes a coulomb matrix and trims it so that only the upper triangular part of the matrix is kept.
        It returns the flattened trimmed array.
        :param X: Coulomb matrix for one sample. numpy array of shape (n_atoms, n_atoms)
        :return: numpy array of shape (n_atoms*(n_atoms+1)/2, )
        """
        size = int(self.n_atoms * (self.n_atoms+1) * 0.5)
        temp = np.zeros((size,))
        counter = 0

        for i in range(self.n_atoms):
            for j in range(i, self.n_atoms):
                temp[counter] = X[i][j]
                counter = counter + 1

        return temp

    def plot(self, X, n=0):
        """
        This function plots a coulomb matrix that is contained in the X descriptor.
        :param X: The coulomb matrix for all samples
        :param n: which line to plot of X
        :return: None
        """
        import seaborn as sns

        matrix = np.reshape(X[n], (self.n_atoms, self.n_atoms))
        sns.heatmap(data=matrix)
        sns.plt.show()



if __name__ == "__main__":

    def testMatrix():
        X = [["H", 0.0, 0.0, 0.0, "H", 1.0, 0.0, 0.0, "C", 0.5, 0.5, 0.5], ["H", 0.1, 0.0, 0.0, "H", 0.9, 0.0, 0.0, "C", -0.5, -0.5, -0.5],
                ["H", -0.1, 0.0, 0.0, "H", 1.1, 0.0, 0.0, "C", 1.0, 1.0, 1.0]]
        y = np.array([4.0, 3.0, 1.0])
        return X, y

    X, y = testMatrix()
    CM = CoulombMatrix(matrixX=X)
    CM.generateES()
    CM.generateSCM()
    X, y = CM.generateRSCM(y, numRep=5)
    X = CM.generateTrimmedCM()

    # CM.plot(X)


