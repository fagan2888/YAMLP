import numpy as np
import ImportData

class PartialCharges():

    def __init__(self, matrixX, matrixY, matrixQ):
        """
        :param matrixX: a list of lists of atom labels and coordinates. size (n_samples, n_atoms*4)
        :param matrixY: a numpy array of energy values of size (N_samples,)
        :param matrixQ: a list of numpy arrays containing the partial charges of each atom. size (n_samples, n_atoms)
        """
        self.rawX = matrixX
        self.rawQ = matrixQ
        self.rawY = matrixY

        self.n_atoms = int(len(self.rawX[0])/4)
        self.n_samples = len(self.rawX)

        self.partQCM = np.zeros((self.n_samples, self.n_atoms ** 2))
        self.partQCM24 = np.zeros((self.n_samples, self.n_atoms ** 2))
        self.diagQ = np.zeros((self.n_samples, self.n_atoms))

    def generatePCCM(self, numRep=1):
        """
        This function generates the new CM that has partial charges instead of the nuclear charges. The diagonal elements
        are q_i^2 while the off diagonal elements are q_i*q_j / R_ij. The descriptor is randomised in the same way as
        the randomly sorted coulomb matrix and becomes an array of size (n_samples*numRep, n_atoms^2)
        :y_data: a numpy array of energy values of size (N_samples,)
        :numRep: number of randomly sorted matrices to generate per sample data - int
        :return: None
        """

        # This is a coulomb matrix for one particular sample in the dataset
        indivPCCM = np.zeros((self.n_atoms, self.n_atoms))
        sampleCount = 0

        for i in range(self.n_samples):
            # Making a vector with the coordinates of the atoms in this data sample
            coord = []
            currentSamp = self.rawX[i]
            for j in range(0,len(currentSamp),4):
                coord.append(np.asarray([currentSamp[j+1], currentSamp[j+2], currentSamp[j+3]]))

            #  Populating the diagonal elements
            for j in range(self.n_atoms):
                indivPCCM[j, j] = self.rawQ[i][j]**2

            # Populating the off-diagonal elements
            for j in range(self.n_atoms - 1):
                for k in range(j + 1, self.n_atoms):
                    # Distance between two atoms
                    distanceVec = coord[j] - coord[k]
                    distance = np.sqrt(np.dot(distanceVec, distanceVec))
                    # Putting the partial charge in
                    indivPCCM[j, k] = self.rawQ[i][j]*self.rawQ[i][k]/ distance
                    indivPCCM[k, j] = indivPCCM[j, k]

            # The partial charge CM for each sample is flattened and added to the total matrix
            self.partQCM[sampleCount, :] = indivPCCM.flatten()
            sampleCount += 1

        self.partQCM, self.y = self.__randomSort(self.partQCM, self.rawY, numRep)

        print "Generated the partial charge coulomb matrix."

    def getPCCM(self):
        """
        Returns the partial charge coulomb matrix and the extended y matrix
        :return: numpy array of size (n_samples*numRep, n_atoms**2), y: extended matrix of energies, numpy array of size
        (n_samples*numRep,)
        """
        return self.partQCM, self.y

    def generatePCCM24(self, numRep=1):
        """
        This function generates the new CM that has partial charges instead of the nuclear charges. The diagonal elements
        are 0.5*q_i^2.4 while the off diagonal elements are q_i*q_j / R_ij.
        :numRep: number of randomly sorted matrices to generate per sample data - int
        :return: None
        """

        # This is a coulomb matrix for one particular sample in the dataset
        indivPCCM = np.zeros((self.n_atoms, self.n_atoms))
        sampleCount = 0

        for i in range(self.n_samples):
            # Making a vector with the coordinates of the atoms in this data sample
            coord = []
            currentSamp = self.rawX[i]
            for j in range(0, len(currentSamp), 4):
                coord.append(np.asarray([currentSamp[j + 1], currentSamp[j + 2], currentSamp[j + 3]]))

            # Populating the diagonal elements
            for j in range(self.n_atoms):
                indivPCCM[j, j] = 0.5 * self.rawQ[i][j] ** 2.4

            # Populating the off-diagonal elements
            for j in range(self.n_atoms - 1):
                for k in range(j + 1, self.n_atoms):
                    # Distance between two atoms
                    distanceVec = coord[j] - coord[k]
                    distance = np.sqrt(np.dot(distanceVec, distanceVec))
                    # Putting the partial charge in
                    indivPCCM[j, k] = self.rawQ[i][j] * self.rawQ[i][k] / distance
                    indivPCCM[k, j] = indivPCCM[j, k]

            # The partial charge CM for each sample is flattened and added to the total matrix
            self.partQCM24[sampleCount, :] = indivPCCM.flatten()
            sampleCount += 1

        self.partQCM24, self.y = self.__randomSort(self.partQCM24, self.rawY, numRep)

        print "Generated the partial charge coulomb matrix (diagonal ^2.4)."

    def getPCCM24(self):
        """
        Returns the partial charge coulomb matrix where the diagonal elements are raised to the 2.4
        :return: numpy array of size (n_samples, n_atoms**2), y: extended matrix of energies, numpy array of size
        (n_samples*numRep,)
        """
        return self.partQCM24, self.y

    def __randomSort(self, X, y, numRep):
        """
        This function randomly sorts the rows of the coulomb matrices depending on their column norm. It generates a
        matrix of size (n_samples*numRep, n_atoms^2)
        :numRep: The number of randomly sorted matrices to be generated for each data sample.
        :return: ranSort: numpy array of size (n_samples*numRep, n_atoms^2),
                y_bigdata: a numpy array of energy values of size (N_samples*numRep,)
        """

        # Checking reasonable numRep value
        if (isinstance(numRep, int) == False):
            print "Error: you gave a non-integer value for the number of RSCM that you want to generate."
            return None
        elif (numRep < 1):
            print "Error: you cannot generate less than 1 RSCM per sample. Enter an integer value > 1."

        counter = 0
        ranSort = np.zeros((self.n_samples * numRep, self.n_atoms ** 2))
        y_bigdata = np.zeros((self.n_samples * numRep,))

        for i in range(self.n_samples):
            tempMat = np.reshape(X[i, :], (self.n_atoms, self.n_atoms))

            # Calculating the norm vector for the coulomb matrix
            rowNorms = np.zeros(self.n_atoms)

            for j in range(self.n_atoms):
                rowNorms[j] = np.linalg.norm(tempMat[j, :])

            for k in range(numRep):
                # Generating random vectors and adding to the norm vector
                randVec = np.random.normal(loc=0.0, scale=np.std(rowNorms), size=self.n_atoms)
                rowNormRan = rowNorms + randVec
                # Sorting the new random norm vector
                permutations = np.argsort(rowNormRan)
                permutations = permutations[::-1]
                # Sorting accordingly the Coulomb matrix
                tempRandCM = tempMat[permutations, :]
                tempRandCM = tempRandCM[:,permutations]
                # Adding flattened randomly sorted Coulomb matrix to the final descriptor matrix
                ranSort[counter, :] = tempRandCM.flatten()
                counter = counter + 1

            # Copying multiple values of the energies
            y_bigdata[numRep * i:numRep * i + numRep] = y[i]

        return ranSort, y_bigdata

    def generateDiagPCCM(self):
        """
        This function generates a descriptor that is a vector of q_i^2. It has the size of (n_samples, n_atoms).
        :return: None
        """

        for i in range(self.n_samples):
            partQ = self.rawQ[i]
            for j in range(self.n_atoms):
                self.diagQ[i][j] = partQ[j]**2

        print "Generated the diagonal of the partial charge coulomb matrix."

    def getDiagPCCM(self):
        """
        This function returns the diagonal of the partial charge coulomb matrix.
        :return: numpy matrix of size (n_sample, n_atoms)
        """
        return self.diagQ

if __name__ == "__main__":
    X, y, q = ImportData.loadPd_q("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/pruning/dataSets/pbe_b3lyp_partQ.csv")
    mat = PartialCharges(X, y, q)
    mat.generatePCCM(numRep=4)