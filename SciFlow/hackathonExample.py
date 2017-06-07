"""
This script uses the new estimator written in tensorflow to fit the data that was used in the hackathon. The
hyperparameters are chosen by me through my knowledge of what worked well in the hackathon.
"""

import CoulombMatrix
import ImportData
import NNFlow
import plotting
import numpy as np
from sklearn import preprocessing as preproc
from sklearn import model_selection as modsel



# Importing the data
X, y = ImportData.loadPd("hackathonData.csv")

# Creating the CM object
coulMat = CoulombMatrix.CoulombMatrix(matrixX=X)
#  Creating the three descriptors
# eigSpec = coulMat.generateES()
# sortMat = coulMat.generateSCM()
ranSorMat, y = coulMat.generateRSCM(y_data=y, numRep=4)

# Normalising the data
X_scal = preproc.StandardScaler().fit_transform(ranSorMat)

# Split into training and test set
X_train, X_test, y_train, y_test = modsel.train_test_split(X_scal, y, test_size=0.1)


# Defining the estimator
estimator = NNFlow.MLPRegFlow(hidden_layer_sizes=(46,),alpha=0.3,learning_rate_init=0.0001,max_iter=500)

# Fitting the model
estimator.fit(X_train,y_train)
estimator.plotTrainCost()

# Calculating the predictions
y_pred = estimator.predict(X_test)

# Correlation plot
plotting.plotSeaborn(y_test, y_pred)

