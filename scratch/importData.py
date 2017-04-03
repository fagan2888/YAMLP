import numpy as np
import os

def XMLtoCSV(XMLinput):
    """ 
    This function takes as an input the XML file that comes out of the electronic structure calculations and transforms 
    it into 2 CSV files. The first one is a 'X matrix of size n_samples x (4 x n_atoms) and the second one is a 
    'Y matrix' of size n_samples x 1 which contains the energy of the configurations stored in the X_matrix. 
    """
    
    # These are the output files
    fileX = open('X.csv', 'w')
    fileY = open('Y.csv', 'w')
    
    # This is the input file
    inputFile = open(XMLinput, 'r')
    
    # The information of the molecule is contained in the block <cml:molecule>...<cml:molecule>.
    # The atom xyz coordinates, the labels and the energy have to be retrieved
    # Each configuration corresponds to one line in the CSV files
    
    for line in inputFile:
        data = []
        if "<cml:molecule>" in line:
            for i in range(3):
                line = inputFile.next()
            while "</cml:atomArray>" not in line:
                indexLab = line.find("elementType=")
                indexX = line.find("x3=")
                indexY = line.find("y3=")
                indexYend = line.find("\n")
                indexZ = line.find("z3=")
                indexZend = line.find("/>")
    
                if indexLab >= 0:
                    data.append(line[indexLab + 13])
                    data.append(line[indexX + 4: indexY - 2])
                    data.append(line[indexY + 4: indexYend - 1])
                if indexZ >= 0:
                    data.append(line[indexZ + 4: indexZend - 1])
    
                line = inputFile.next()
            for i in range(len(data)):
                fileX.write(data[i])
                fileX.write(",")
            fileX.write("\n")
    
        if '<property name="Energy"' in line:
            line = inputFile.next()
            indexEn1 = line.find("value")
            indexEn2 = line.find("/>")
            energy = line[indexEn1+7:indexEn2-1]
            fileY.write(energy + "\n")

    return None

def XYZtoCSV(XYZinput):
    """
    This function takes as an input the XYZ file that comes out of VR and transforms
    it into 2 CSV files. The first one is a 'X matrix of size n_samples x (4 x n_atoms) and the second one is a
    'Y matrix' of size n_samples x 1 which contains the energy of the configurations stored in the X_matrix.

    VERY SPECIFIC FUNCTION - NOT GENERAL!!!
    """

    # These are the output files
    fileX = open('X.csv', 'w')
    fileY = open('Y.csv', 'w')

    # This is the input file
    inputFile = open(XYZinput, 'r')

    isFirstLine = True
    n_atoms = 0

    for line in inputFile:
        if isFirstLine:
            n_atoms = int(line)
            isFirstLine = False

        index1 = line.find("Energy")
        if index1 >= 0:
            index2 = line.find("(hartree)")
            energyHa = float(line[index1+8:index2-1])
            energyKjmol = energyHa * 2625.4988
            fileY.write(str(energyKjmol))
            fileY.write("\n")

        if line[0] == "C" or line[0] == "H":
            line = line.replace("\n", "")
            line = line.replace("\t",",")
            fileX.write(line)
            fileX.write(",")

        if line[0] == "N":
            line = line.replace("\n", "")
            line = line.replace("\t", ",")
            fileX.write(line)
            fileX.write("\n")

def CCdistance():
    """
    This function creates a csv file with the Carbon Carbon distance.
    :return: None
    VERY-SPECIFIC FUNCTION. NOT GENERAL
    """
    input = open("X.csv", "r")
    output = open("Z.csv", "w")

    for line in input:
        lineList = line.split(",")

        # Calculating CC distance
        c1 = lineList[1:4]
        c2 = lineList[21:24]

        for i in range(3):
            c1[i] = float(c1[i])
            c2[i] = float(c2[i])

        c1pos = np.asarray(c1)
        c2pos = np.asarray(c2)

        cc_dist_vec = c2pos - c1pos
        cc_dist = np.sqrt(np.dot(cc_dist_vec, cc_dist_vec))

        # Calculating the shortest NH distance
        h1 = lineList[5:8]
        h2 = lineList[9:12]
        h3 = lineList[13:16]
        h4 = lineList[17:20]
        n = lineList[25:]

        for i in range(3):
            n[i] = float(n[i])

        n = np.asarray(n)
        hList = [h1, h2, h3, h4]
        nhDistances = []

        for item in hList:
            for i in range(3):
                item[i] = float(item[i])

            item = np.asarray(item)
            item = n - item
            nhDist = np.sqrt(np.dot(item, item))
            nhDistances.append(nhDist)

        min_dist = min(nhDistances)

        output.write(str(cc_dist) + "," + str(min_dist))
        output.write("\n")



    input.close()
    output.close()

def loadY(fileY):
    """
    This function takes as input the CSV file for the Y part of the data and
    returns a numpy array of size (n_samples, 1)
    """

    # Checking that the input file has the correct .csv extension
    if fileY[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(fileY, 'r')

    y_list = []
    for line in inputFile:
        y_list.append(float(line))

    matrixY = np.asarray(y_list).reshape((len(y_list), 1))

    inputFile.close()
    #os.remove(fileY)

    return matrixY

def loadX(fileX):
    """
    This function takes as input the CSV file for the X part of the data and
    returns a list of lists of size (n_samples, n_features)
    """

    if fileX[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(fileX, 'r')

    # Creating an empty matrix of the right size
    matrixX = []

    for line in inputFile:

        line = line.replace(",\n","")
        listLine = line.split(",")

        # converting the numbers to float
        for i in range(0,len(listLine)-1,4):
            for j in range(3):
                listLine[i+j+1] = float(listLine[i+j+1])
        matrixX.append(listLine)


    inputFile.close()
    #os.remove(fileX)

    return matrixX

def loadZ(fileZ):
    """
        This function takes as input the CSV file for the Y and Z part of the data and
        returns a numpy array of size (n_samples, n_features)
        """

    # Checking that the input file has the correct .csv extension
    if fileZ[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(fileZ, 'r')

    y_list = []
    z_list = []
    for line in inputFile:
        line = line.replace("\n", "")
        listLine = line.split(",")
        y_list.append(float(listLine[0]))
        z_list.append(float(listLine[1]))

    matrixY = np.asarray(y_list).reshape((len(y_list), 1))
    matrixZ = np.asarray(z_list).reshape((len(z_list), 1))

    inputFile.close()
    # os.remove(fileY)

    return matrixY, matrixZ

def splitData(X, Y, percentages):
    """
    :param X: This is the X descriptor of the system and it is a np matrix of size (n_samples, n_features)
    :param Y: This is the Y corresponding to the descriptor. It is a np array of size (n_samples, 1)
    :param percentages: This is a numpy array of 3 values between 0 and 1 that specify the proportions of data to put in the
                        training, cross-validation and validation set respectively.
    :return: It returns two lists of split data, one for the X descriptor and one for the Y values.
    """
    np.random.shuffle(X)
    np.random.shuffle(Y)

    n_samples = Y.shape[0]
    setSizes = n_samples * percentages
    setSizes = setSizes.astype(int)

    # This checks that the sum of the number of samples in each set matches the total number of samples
    # if the number of samples is larger than the total number of samples, some samples are removed from the training
    # dataset. In the opposite case, they are added to the training dataset.
    if np.sum(setSizes) > n_samples:
        setSizes[0] -= (np.sum(setSizes) - n_samples)
    elif np.sum(setSizes) < n_samples:
        setSizes[0] += (n_samples - np.sum(setSizes))

    # Taking the first part of X and Y for the training set
    X_train = X[0:setSizes[0], :]
    Y_train = Y[0:setSizes[0], :]

    X_crossVal = X[setSizes[0]:(setSizes[1] + setSizes[0]), :]
    Y_crossVal = Y[setSizes[0]:(setSizes[1] + setSizes[0]), :]

    X_val = X[(setSizes[1] + setSizes[0]):np.sum(setSizes), :]
    Y_val = Y[(setSizes[1] + setSizes[0]):np.sum(setSizes), :]

    splitX = [X_train, X_crossVal, X_val]
    splitY = [Y_train, Y_crossVal, Y_val]

    return splitX, splitY

def interpolData(X, Y, numPoints):
    """
    This function takes the X matrix ( list of lists of size (n_samples, n_features) - labels of atoms and their
    coordinate ) and for each atom coordinate of two samples it creates samples that have coordinates in between
    the coordinates of the two initial atoms. The same is done for the energy.
    :param X: list of lists of size (n_samples, n_features) - labels of atoms and their coordinate )
    :param Y: numpy array of size (n_samples, 1)
    :param numPoints: Number of samples to create via interpolation. numPoints = 10 creates 8 new points...
    :return: it returns a new extended X and Y.
    """

    addSamplesX = []
    addSamplesY = []
    addSamplesAng = []

    for i in range(0,len(X)-1):

        newSamp = [[] for _ in range(numPoints)]

        # Interpolate the energies and angles
        e1 = Y[i,0]
        e2 = Y[i+1,0]
        a1 = Y[i,1]
        a2 = Y[i+1,1]

        newEne = np.arange(e1, e2, ((e2 - e1) / numPoints))
        newAng = np.arange(a1, a2, ((a2 - a1) / numPoints))

        # Go through the coordinates the atoms one by one in the sample
        for k in range(0, len(X[i]), 4):

            # X, y and z values of the interpolated coordinates (including the initial and end point)
            newX = np.arange(X[i][k+1], X[i + 1][k+1], ((X[i + 1][k+1] - X[i][k+1]) / (numPoints+2)))
            newY = np.arange(X[i][k+2], X[i + 1][k+2], ((X[i + 1][k+2] - X[i][k+2]) / (numPoints+2)))
            newZ = np.arange(X[i][k+3], X[i + 1][k+3], ((X[i + 1][k+3] - X[i][k+3]) / (numPoints+2)))

            # Arranging the data in the 'atom-label x y z' format one atom at a time
            for j in range(numPoints):
                newSamp[j].append(X[i][k])
                newSamp[j].append(newX[j+1])
                newSamp[j].append(newY[j+1])
                newSamp[j].append(newZ[j+1])

        # Adding each formatted list into the final list containing all the interpolated points
        for l in range(numPoints):
            addSamplesX.append(newSamp[l])
            addSamplesY.append(newEne[l])
            addSamplesAng.append(newAng[l])

    finalEn = (np.asarray(addSamplesY)).reshape((len(addSamplesY), 1))
    finalAng = (np.asarray(addSamplesAng)).reshape((len(addSamplesAng), 1))
    finalY = np.concatenate((finalEn, finalAng), axis=1)

    return addSamplesX, finalY






if __name__ == "__main__":
    XMLtoCSV("/Users/walfits/Repositories/tensorflow/AMP/input1.xml")

    X_total = loadX("X.csv")
    Y_total = loadY("Y.csv")

    interpolData(X_total, Y_total, 2)