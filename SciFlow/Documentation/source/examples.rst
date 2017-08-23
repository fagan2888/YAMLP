Example 1: Training a model with the Coulomb matrix
****************************************************

In order to make this example work, import the following modules::

    import ImportData
    import CoulombMatrix
    import NNFlow
    import CoulombMatrix
    from sklearn import model_selection as modsel
    import pickle
    import os.path

Then, use the data set that can be found `here <https://github.com/SilviaAmAm/trainingNN/blob/master/dataSets/PBE_B3LYP/pbe_b3lyp_partQ_rel.csv/>`_. This data set contains the xyz coordinates of each configuration, the partial charge of the atoms and the energies at the PBE and the B3LYP level. In order to import the data, use::

    X, y, Q = ImportData.loadPd_q("pbe_b3lyp_partQ_rel.csv")

Then, to generate the partially randomised Coulomb matrix with nuclear charges::

    CM = CoulombMatrix.CoulombMatrix(matrixX=X)
    X_cm, y_cm = CM.generatePRCM(y, numRep=5)

This will do 5 randomisations of each sample. Then, one can use Scikit learn to randomly split the data into a training set and a training set::

    X_train, X_test, y_train, y_test = modsel.train_test_split(X_cm, y_cm, test_size=0.2)

Then, to train the model::

    # Defining the estimator
    estimator = NNFlow.MLPRegFlow(max_iter=3100, batch_size=550, alpha=0.0001, learning_rate_init=0.0002, hidden_layer_sizes=(18,))

    # Training the neural net
    estimator.fit(X_train, y_train, X_test, y_test)
    estimator.plotLearningCurve()
    estimator.errorDistribution(X_test, y_test)``

The first line generates the estimator and sets some hyperparameters. Then, the model is fit to the X_train and y_train data. Two curves showing the square error evolution for the training and the test sets are plotted. Then, a distribution of the square error is also plotted.

To score fully the model, the following can be done::

    r2, rmse, mae, lpo, lno = estimator.scoreFull(X_test, y_test)
    print "On test set:"
    print "R^2: " + str(r2)
    print "rmse (kJ/mol): " + str(rmse * 2625.50)
    print "mae (kJ/mol): " + str(mae * 2625.50)
    print "The largest positive outlier (kJ/mol): " + str(lpo * 2625.50)
    print "The largest negative outlier (kJ/mol): " + str(lno * 2625.50)

    # Calculating the predictions
    y_pred = estimator.predict(X_test)

    # Correlation plot
    estimator.correlationPlot(X_test, y_test, ylim=(-0.075,0.040), xlim=(-0.075,0.040))``


