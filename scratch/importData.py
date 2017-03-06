import numpy as np

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


def loadY(YFile):
    """
    This function takes as input the CSV file for the Y part of the data and
    returns a numpy array of size (n_samples, 1)
    """


    # Checking that the input file has the correct .csv extension
    if YFile[-4:] != ".csv":
        print "Error: the file extension is not .csv"
        quit()

    inputFile = open(YFile, 'r')

    y_list = []
    for line in inputFile:
        y_list.append(float(line))

    matrixY = np.asarray(y_list).reshape((len(y_list), 1))

    inputFile.close()

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

        listLine = line.split(",")

        # converting the numbers to float
        for i in range(0,len(listLine)-1,4):
            for j in range(3):
                listLine[i+j+1] = float(listLine[i+j+1])
        matrixX.append(listLine[0:-1])

    inputFile.close()

    return matrixX


if __name__ == "__main__":
    X = loadX("X.csv")
    print X