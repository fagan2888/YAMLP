from ase import Atoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork

### This turns the X.csv and Y.csv into a list of ASE atoms:

fileX = open("X.csv", "r")
fileY = open("Y.csv", "r")

samples = []        # List of ASE Atoms objects
n_samples = 0       # number of training samples

for line in fileX:
    listLine = line.split(",")
    labels = ""         # This is a string containing all the atom labels
    coord = []          # This is a list of tuples with the coordinates of all the atoms in a configuration

    for i in range(0,len(listLine)-1,4):
        labels = labels + listLine[i]
        coord.append( (float(listLine[i+1]), float(listLine[i+2]), float(listLine[i+3])) )

    config = Atoms(labels, coord)       # This makes an Atoms object for each sample in the csv file
    samples.append(config)
    n_samples += 1



