import numpy as np
import ImportData

class tewDescriptor:

    def __init__(self, matrixX):
        self.rawX = matrixX
        self.n_atoms = int(len(self.rawX[0]) / 4)
        self.n_samples = len(self.rawX)
        self.n_distances = int(self.n_atoms * (self.n_atoms - 1) * 0.5)
        self.tew = np.zeros((self.n_samples,self.n_distances))

    def generateTew(self):

        for i in range(self.n_samples):
            # Making a vector with the coordinates of the atoms in this data sample
            counter = 0
            coord = []
            currentSamp = self.rawX[i]
            for j in range(0, len(currentSamp), 4):
                coord.append(np.asarray([currentSamp[j + 1], currentSamp[j + 2], currentSamp[j + 3]]))

            for j in range(0,self.n_atoms-1):
                for k in range(j+1, self.n_atoms):
                    distVec = coord[j] - coord[k]
                    dist = np.sqrt(np.dot(distVec, distVec))

                    self.tew[i,counter] = dist
                    counter = counter + 1

    def getTew(self):
        """
        This function returns the tew descriptor.
        :return: a numpy array of size (n_samples, 0.5 * n_atoms * (n_atoms-1))
        """
        return self.tew

if __name__ == "__main__":
    X, y, q = ImportData.loadPd_q(
        "/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/pruning/dataSets/pbe_b3lyp_partQ.csv")
    desc = tewDescriptor(X)
    desc.generateTew()