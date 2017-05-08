""" This script prunes the raw data that came out of the CH4+CN experiment in VR """


import importData
import numpy as np
import matplotlib.pyplot as plt



def plotEneDistribution(energies, numIntervals):

    """
    This function looks at the file containing all the energies and plots their frequency
    energies: numpy array of all the energies, of size (n_samples, 1)
    numIntervals: The number of binds in which to divide the energies
    """

    # Reshaping the energies

    energies = np.reshape(energies, (energies.shape))

    # Finding the max and min value to generate a list of bins

    maxEne = np.amax(energies)
    minEne = np.amin(energies)
    interval = (maxEne - minEne) / numIntervals
    bins = np.arange(minEne, maxEne, interval)
    histog, bin_edges = np.histogram(energies, bins)

    # Plotting the distribution

    plt.plot(bins[1:], histog)
    plt.xlabel('Energy (kJ/mol)')
    plt.ylabel('Number of occurrences')
    plt.show()

def cutOutliars(eneRange, energies):
    outX = open('Xpruned1.csv', 'w')
    outY = open('Ypruned1.csv', 'w')
    inX = open('X.csv', 'r')

    counter = 0

    for line in inX:
        if (energies[counter, 0] > eneRange[0]) and (energies[counter, 0] < eneRange[1]):
            outX.write(line)
            outY.write(str(energies[counter,0]) + "\n")

        counter = counter + 1

    outX.close()
    outY.close()
    inX.close()

if __name__ == "__main__" :

    importData.XYZtoCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/combinedTraj.xyz")
    energies = importData.loadY("Y.csv")
    plotEneDistribution(energies, 500)

    eneRange = [-19000, -18700]
    cutOutliars(eneRange, energies)

    energiesPruned = importData.loadY("Ypruned1.csv")
    plotEneDistribution(energiesPruned, 500)

