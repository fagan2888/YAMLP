import ImportData as imdat
import CoulombMatrix
import PartialCharge
from sklearn import preprocessing as preproc
import numpy as np

X, y, Q = imdat.loadPd_q('dataSets/pbe_b3lyp_partQ_rel.csv')

# Generating coulomb matrix
descr1 = CoulombMatrix.CoulombMatrix(matrixX=X)
X_coul, y_coul = descr1.generateRSCM(y_data=y, numRep=4)
X_coul_scal = preproc.StandardScaler().fit_transform(X_coul)
y_coul = np.reshape(y_coul, (len(y_coul), 1))

# Generating the Partial Charge Coulomb matrix (diagonal elements are q_i^2 while the off diagonal elements
# are q_i*q_j / R_ij)
descr2 = PartialCharge.PartialCharges(X, y, Q)
descr2.generatePCCM(numRep=4)
X_pccm, y_pccm = descr2.getPCCM()
X_pccm_scal = preproc.StandardScaler().fit_transform(X_pccm)
y_pccm = np.reshape(y_pccm, (len(y_pccm), 1))

# Generating the second Partial charge coulomb matrix (diagonal elements are 0.5*q_i^2.4 while the off diagonal
# elements are q_i*q_j / R_ij)
descr3 = PartialCharge.PartialCharges(X, y, Q)
descr3.generatePCCM24(numRep=4)
X_pccm24, y_pccm24 = descr3.getPCCM24()
X_pccm24_scal = preproc.StandardScaler().fit_transform(X_pccm24)
y_pccm24 = np.reshape(y_pccm24, (len(y_pccm24), 1))

# Concatenating the X and y to output in the format that osprey wants
data_coul = np.append(X_coul_scal, y_coul, axis=1)
data_pccm = np.append(X_pccm_scal, y_pccm, axis=1)
data_pccm24 = np.append(X_pccm24_scal, y_pccm24, axis=1)

# Saving datasets to files
outfile1 = open('dataSets/data_coul.csv', 'w')
outfile2 = open('dataSets/data_pccm.csv', 'w')
outfile3 = open('dataSets/data_pccm24.csv', 'w')

np.savetxt(outfile1, data_coul, delimiter=',')
np.savetxt(outfile2, data_pccm, delimiter=',')
np.savetxt(outfile3, data_pccm24, delimiter=',')




