import numpy as np
import ImportData

class PartialCharges():

    def __init__(self, matrixX, matrixQ):
        self.rawX = matrixX
        self.rawQ = matrixQ

        self.n_atoms = int(len(self.rawX[0])/4)
        self.n_samples = len(self.rawX)

    def generatePCCM(self):
        """
        This function generates the new CM that has partial charges instead of the nuclear charges. The diagonal elements
        are q_i^2 while the off diagonal elements are q_i*q_j / R_ij.
        :return: None
        """
        self.partQCM = np.zeros((self.n_samples, self.n_atoms ** 2))

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

        print "Generated the partial charge coulomb matrix."

    def getPCCM(self):
        """
        Returns the partial charge coulomb matrix
        :return: numpy array of size (n_samples, n_atoms**2)
        """
        return self.partQCM

    def generatePCCM24(self):
        """
        This function generates the new CM that has partial charges instead of the nuclear charges. The diagonal elements
        are 0.5*q_i^2.4 while the off diagonal elements are q_i*q_j / R_ij.
        :return: None
        """
        self.partQCM24 = np.zeros((self.n_samples, self.n_atoms ** 2))

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

        print "Generated the partial charge coulomb matrix (diagonal ^2.4)."

    def getPCCM24(self):
        """
        Returns the partial charge coulomb matrix where the diagonal elements are raised to the 2.4
        :return: numpy array of size (n_samples, n_atoms**2)
        """
        return self.partQCM24




if __name__ == "__main__":
    X, y, q = ImportData.loadPd_q("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/pruning/dataSets/pbe_b3lyp_partQ.csv")
    mat = PartialCharges(X, q)
    mat.generatePCCM24()