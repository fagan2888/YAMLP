import numpy as np

def getEnergy(fileY):
    energiesList = []
    for line in fileY:
        energiesList.append(float(line))

    energies = np.asarray(energiesList).reshape((len(energiesList), 1))
    return energies