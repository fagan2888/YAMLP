import numpy as np
import os

def XMLtoCSV(XMLinput):
    """
    This function takes as an input the XML file that comes out of the electronic structure calculations and transforms
    it into 2 CSV files. The first one is the 'X part' of the data. It contains a sample per line. Each line has a format:
    atom label (string), coordinate x (float), coordinate y (float), coordinate z (float), ... for each atom in the system.
    The second file contains the 'Y part' of the data. It has a sample per line with the energy of each sample (float).

    :XMLinput: an XML file obtained from grid electronic structure calculations
    :returns: None
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
            energyHa = float(line[indexEn1 + 7:indexEn2 - 1])
            energyKjmol = energyHa * 2625.4988
            fileY.write(str(energyKjmol) + "\n")

    return None

def XYZtoCSV(XYZinput):
    """
    This function takes as an input the XYZ file that comes out of VR and transforms
    it into 2 CSV files. The first one is the 'X part' of the data. It contains a sample per line. Each line has a format:
    atom label (string), coordinate x (float), coordinate y (float), coordinate z (float), ... for each atom in the system.
    The second file contains the 'Y part' of the data. It has a sample per line with the energy of each sample (float).

    :XMLinput: an XML file obtained from grid electronic structure calculations
    :returns: None

    Note: This is specific to a file containing C, H, N as the atoms.
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

def extractMolpro(MolproInput):
    """
    This function takes a Molpro .out file and returns the geometry, the energy and the partial charges on the atoms.

    :param MolproInput: the molpro .out file
    :return: rawData: List of strings with atom label and atom coordinates - example ['C', '0.1, '0.1', '0.1', ...]
            ene: Value of the energy: string
            partialCh: List of strings with atom label and its partial charge - example ['C', '6.36', 'H', ...]
    """

    # This is the input file
    inputFile = open(MolproInput, 'r')

    # This will contain the data
    rawData = []
    ene = "0"
    partialCh = []


    for line in inputFile:
        # The geometry is found on the line after the keyword "geometry={"
        if "geometry={" in line:
            for i in range(7):
                line = inputFile.next()
                line = line.strip()
                lineSplit = line.split(" ")
                for j in range(len(lineSplit)):
                    rawData.append(lineSplit[j])
        # The energy is found two lines after the keyword "Final beta  occupancy:"
        elif "Final beta  occupancy:" in line:
            line = inputFile.next()
            line = inputFile.next()
            line = line.strip()
            ene = line[len("!RKS STATE 1.1 Energy"):].strip()
        elif "Total charge composition:" in line:
            line = inputFile.next()
            line = inputFile.next()
            for i in range(7):
                line = inputFile.next()
                lineSplit = line.rstrip().split(" ")
                lineSplit = filter(None, lineSplit)
                partialCh.append(lineSplit[1])
                partialCh.append(lineSplit[-2])

    return rawData, ene, partialCh

def list_files(dir, key):
    """
    This function walks through a directory and makes a list of the files that have a name containing a particular string

    :param dir: path to the directory to explore
    :param key: string to look for in file names
    :return: list of files containing "key" in their filename
    """

    r = []  # List of files to be joined together
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]

        for file in files:
            isTrajectory = file.find(key)
            if isTrajectory >= 0:
                r.append(subdir + "/" + file)
    return r

def MolproToCSV(directory, key):
    """
    This function extracts all the geometries and energies from Molpro .out files contained in a particular directory.
    Only the files that have a particular string in their filename will be read. The geometries are then written to X.csv
    where each line is a different geometry. The energies are written to Y.csv where each line is the energy of a
    different geometry. The partial charges are written to partialCh.csv

    :param directory: path to the directory containing the Molpro .out files
    :param key: string to look for in the file names
    :return: None
    """

    # These are the output files
    fileX = open('X.csv', 'w')
    fileY = open('Y.csv', 'w')
    fileZ = open('partialCh.csv', 'w')

    # Obtaining the list of files to mine
    fileList = list_files(directory, key)

    # Iterating over all the files
    for item in fileList:
        # Extracting the geometry and the energy from a Molpro out file
        geom, ene, partialCh = extractMolpro(item)
        if len(geom) != 28 or ene == "0" or len(partialCh) != 14:
            print "The following file couldn't be read properly:"
            print item + "\n"
            continue
        for i in range(len(geom)):
            fileX.write(geom[i])
            fileX.write(",")
        fileX.write("\n")
        fileY.write(ene + "\n")

        for i in range(len(partialCh)):
            fileZ.write(partialCh[i])
            fileZ.write(",")
        fileZ.write("\n")

def loadX(fileX):
    """
    This function takes a .csv file that contains on each line a different configuration of the system in the format
    "C,0.1,0.1,0.1,H,0.2,0.2,0.2..." and returns a list of lists with the configurations of the system.

    :param fileX: The .csv file containing the geometries of the system
    :return: a list of lists with characters and floats. An example is shown below for 3 samples of a di hydrogen
            molecule:
        [['H',-0.5,0.0,0.0,'H',0.5,0.0,0.0], ['H',-0.3,0.0,0.0,'H',0.3,0.0,0.0], ['H',-0.7,0.0,0.0,'H',0.7,0.0,0.0]]
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
    return matrixX

def loadY(fileY):
    """
    This function takes a .csv file containing the energies of a system and returns an array with the energies contained
    in the file.

    :param fileY: the .csv file containing the energies of the system
    :return: matrixY - a numpy array containing the energies of the system with size (n_samples, 1)
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
    return matrixY

def loadPd(fileName):
    """
    This function takes a .csv file that contains on each line a different configuration of the system in the format
    "C,0.1,0.1,0.1,H,0.2,0.2,0.2..." and at the end of each line there are two values of the energies. It returns a
    list of lists with the configurations of the system and a numpy array of size (N_samples, 1) with the difference of
    the two values of the energies.

    :param fileX: The .csv file containing the geometries and the energies at 2 levels of theory for the system
    :return: a list of lists with characters and floats. An example is shown below for 3 samples of a di hydrogen
            molecule:
        [['H',-0.5,0.0,0.0,'H',0.5,0.0,0.0], ['H',-0.3,0.0,0.0,'H',0.3,0.0,0.0], ['H',-0.7,0.0,0.0,'H',0.7,0.0,0.0]]
        and a list of energy differences of size (n_samples,)
    """
    if fileName[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(fileName, 'r')

    # Creating a matrix with the raw data:
    rawData = []
    matrixX = []
    matrixY = []

    isFirstLine = True

    for line in inputFile:
        if isFirstLine == True:
            line = inputFile.next()
            isFirstLine = False

        line = line.replace("\n","")
        listLine = line.split(",")

        ene = listLine[-2:]
        geom = listLine[1:-2]

        for i in range(len(ene)):
            ene[i] = float(ene[i])

        eneDiff = ene[0] - ene[1]
        matrixY.append(eneDiff)

        for i in range(0,len(geom)-1,4):
            for j in range(3):
                geom[i+j+1] = float(geom[i+j+1])
        matrixX.append(geom)

    matrixY = np.asarray(matrixY)
    inputFile.close()

    return matrixX, matrixY

def loadPd_q(fileName):
    """
    This function takes a .csv file containing a dataset formatted with the geometries in a 'clean datases' arrangement,
    followed by the partial charges and 2 values of the energies.
    :param fileName: .csv file
    :return: matrixX: a list of lists with characters and floats. An example is shown below for 3 samples of a di hydrogen
            molecule:
            [['H',-0.5,0.0,0.0,'H',0.5,0.0,0.0], ['H',-0.3,0.0,0.0,'H',0.3,0.0,0.0], ['H',-0.7,0.0,0.0,'H',0.7,0.0,0.0]]
            matrixY: a numpy array of energy differences (floats) of size (n_samples,)
            matrixQ: a list of numpy arrays of the partial charges -  size (n_samples, n_atoms)
    """

    if fileName[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(fileName, 'r')
    isFirstLine = True

    # Lists that will contain the data
    matrixX = []
    matrixY = []
    matrixQ = []

    # Reading the file
    for line in inputFile:
        if isFirstLine:
            line = inputFile.next()
            isFirstLine = False

        line = line.replace("\n", "")
        listLine = line.split(",")

        geom = extractGeom(listLine)
        eneDiff = extractEneDiff(listLine)
        partQ = extractQ(listLine)

        matrixX.append(geom)
        matrixY.append(eneDiff)
        matrixQ.append(partQ)

    matrixY = np.asarray(matrixY)

    return matrixX, matrixY, matrixQ

def extractGeom(lineList):
    """
    Function used by loadPd_q to extract the geometries.
    :param lineList: line with geometries in clean format, partial charges and energies
    :return: list of geometry in format [['H',-0.5,0.0,0.0,'H',0.5,0.0,0.0], ['H',-0.3,0.0,0.0,'H',0.3,0.0,0.0]...
    """
    geomPart = lineList[1:22]
    atomLab = ["C","H","H","H","H","C","N"]
    finalGeom = []

    for i in range(len(atomLab)):
        finalGeom.append(atomLab[i])
        for j in range(3):
            finalGeom.append(float(geomPart[3*i+j]))

    return finalGeom

def extractEneDiff(lineList):
    """
    Extracts the energy from a line of the clean data set
    :param lineList: line with geometries in clean format, partial charges and energies
    :return: energy difference (float)
    """
    enePart = lineList[-2:]
    eneDiff = float(enePart[0]) - float(enePart[1])
    return eneDiff

def extractQ(lineList):
    """
    Extracts the partial charges from a line of the clean data set
    :param lineList: line with geometries in clean format, partial charges and energies
    :return: numpy array of partial charges of size (n_atoms)
    """
    qPart = lineList[22:-2]
    for i in range(len(qPart)):
        qPart[i] = float(qPart[i])

    qPart = np.asarray(qPart)
    return qPart