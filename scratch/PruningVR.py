""" This script prunes the raw data that came out of the CH4+CN experiment in VR """


import importData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



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
    plt.xlabel('Energy (Ha)')
    plt.ylabel('Number of occurrences')
    plt.show()

def cutOutliars(eneRange, energies):
    """This function takes a set of energies from different configurations and an energy range. It goes then through all
     the energies and checks whether they are in a certain energy range. If they are, their corresponding chemical
     configuration is added to a file of pruned structures and the energies are added to a file of pruned energies.
     Input:
     eneRange: array of a low and high bound to the energy range. Configurations with energies outside this range will removed.
     energies: numpy array of energies from electronic structure calculations.
     """


    outX = open('Xpruned1.csv', 'w')
    outY = open('Ypruned1.csv', 'w')
    outZ = open("Zpruned1.csv", "w")
    inX = open('X.csv', 'r')
    inZ = open("Z.csv", "r")

    counter = 0

    dim = energies.shape

    # This list contains a 1 if a configuration is to be kept and a 0 if it is to be removed
    toPrune = []

    for i in range(dim[0]):
        if (energies[i, 0] > eneRange[0]) and (energies[i, 0] < eneRange[1]):
            toPrune.append(1)
        else:
            toPrune.append(0)

    for line in inX:
        if toPrune[counter] == 1:
            outX.write(line)
            outY.write(str(energies[counter,0]) + "\n")

        counter = counter + 1

    counter = 0
    for line in inZ:
        if toPrune[counter] == 1:
            outZ.write(line)

        counter = counter + 1

    outX.close()
    outY.close()
    outZ.close()
    inX.close()
    inZ.close()



def plotDistances(x, y, x_label):
    fig, ax = plt.subplots(figsize=(8,7))
    ax.scatter(x, y, marker="o", c="r", edgecolor="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel('Energy (Ha)')
    plt.show()

def plotDist_seaborn(x, y):
    """
    Plots y as a function of x
    :param x: numpy array of size (n_sample, n_features)
    :param y: numpy array of size (n_sample, n_features)
    :return: None
    """
    y = y*2625

    df = pd.DataFrame()
    df['CC distance (A)'] = x[:,0]
    df['Energy (kJ/mol)'] = y[:,0]
    sns.lmplot('CC distance (A)','Energy (kJ/mol)', data=df, scatter_kws={"s": 10, "alpha": 0.4}, fit_reg=False)
    sns.plt.show()


if __name__ == "__main__" :

    ### To import the DFTB data
    # importData.XYZtoCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/pruning/level0/combinedTraj.xyz")
    ### To import the PBE data
    importData.MolproToCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/pruning/level1_Molpro", "3g-u.out")


    energies = importData.loadY("Y.csv")
    importData.CCdistance("X.csv")
    distance = importData.loadZ("Z.csv")

    eneRange = [-19000, -18700]
    cutOutliars(eneRange, energies)

    energiesPruned = importData.loadY("Ypruned1.csv")
    distancePruned = importData.loadZ("Zpruned1.csv")


    # plotDist_seaborn(distancePruned[0], energiesPruned)
    plotDist_seaborn(distance[0], energies)


    # plotDistances(distance[0], energies, "CC distance")
    # plotDistances(distance[1], energies, "min NH distance")
    # plotDistances(distance[2], energies, "min CH distance")
    # plotDistances(distance[3], energies, "max CN distance")






