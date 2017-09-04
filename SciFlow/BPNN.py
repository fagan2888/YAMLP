"""
This module implements a Behler-Parinello Neural network that is compatible with Scikit learn and can therefore be
used with Osprey hyperparameter optimisation.

This code was written following closely the code written by Zachary Ulissi (Department of Chemical Engineering,
Stanford University) in the tflow.py module of the AMP package.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class BPNN(BaseEstimator, ClassifierMixin):
    """
    The parameter labels says to which atom each feature in X corresponds to. It is a list of atom label plus the number
    of how many features correspond to that atom.

    :labels: list of length (2*n_atoms,) containing strings and int
    """
    def __init__(self, hidden_layer_sizes=(5,), alpha=0.0001, batch_size='auto', learning_rate_init=0.001,
                 max_iter=80, labels=(0,)):

        # Initialising the parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        # Terrible coding, need to find a better solution to this
        if labels == (0,):
            self.labels = [('N', 4), ('C', 4), ('C', 4), ('H', 4), ('H', 4), ('H', 4), ('H', 4)]
        else:
            self.labels = labels

    def fit(self, X, y):
        """
        X is the training set. It is a numpy array containing all the descriptors for each atom concatenated.

        :X: numpy array of shape (n_samples, n_features)
        :y: array of shape (n_samples,)
        """
        # Some useful data
        self.n_samples = X.shape[0]
        self.checkBatchSize()

        # Modifying shape of y to be compatible with tensorflow and creating a placeholder
        y = np.reshape(y, (len(y), 1))

        with tf.name_scope('input_y'):
            y_tf = tf.placeholder(tf.float32, [None, 1])

        # # Splitting the X into the different descriptors
        # X_input = self.__split_input(X)

        # Making a list of the unique elements and one of all the elements in order
        self.unique_ele, self.all_atoms = self.__unique_elements()

        # Create a list of tensorflow placeholders, one item per atom in the system
        inputs = []

        with tf.name_scope('input_x'):
            for ii in range(0, len(self.labels)):
                inputs.append(tf.placeholder(tf.float32, [None, self.labels[ii][1]]))

        # Zipping the placeholders and the labels, so it is easy to know which weights to call
        data = zip(self.all_atoms, inputs)

        # Declaring the weights
        with tf.name_scope('weights'):
            all_weights = {}
            all_biases = {}

            for key, value in self.unique_ele.iteritems():
                weights, biases = self.__generate_weights(value)
                all_weights[key] = weights
                all_biases[key] = biases

                tf.summary.histogram("weights_in", weights[0])
                for ii in range(len(self.hidden_layer_sizes) - 1):
                    tf.summary.histogram("weights_hidden", weights[ii + 1])
                tf.summary.histogram("weights_out", weights[-1])

        # Evaluating the model
        with tf.name_scope("atom_nn"):
            all_atom_ene = []
            for ii in range(len(self.all_atoms)):
                atom_ene = self.__atom_energy(data[ii], all_weights, all_biases)
                all_atom_ene.append(atom_ene)

        # Summing the results to get the total energy
        with tf.name_scope("tot_ene"):
            model_tot = all_atom_ene[0]
            for ii in range(len(all_atom_ene)-1):
                model_tot = tf.add(all_atom_ene[ii+1], model_tot)

        # Calculating the cost function with L2 regularisation term
        with tf.name_scope('cost'):
            cost = self.__reg_cost(model_tot, y_tf, all_weights)
            # tf.summary.scalar("cost", cost)

        # Training step
        with tf.name_scope('training'):
            optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

        # Initialisation of the model
        init = tf.global_variables_initializer()
        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            self.cost_list = []
            summary_writer = tf.summary.FileWriter(logdir="/Users/walfits/Repositories/trainingNN/tensorboard", graph=sess.graph)
            sess.run(init)

            for i in range(self.max_iter):
                # This is the total number of batches in which the training set is divided
                n_batches = int(self.n_samples / self.batch_size)
                # This will be used to calculate the average cost per iteration
                avg_cost = 0
                # Learning over the batches of data
                for i in range(n_batches):
                    batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
                    X_batch = self.__split_input(batch_x)
                    feeddict = {i: d for i, d in zip(inputs, X_batch)}
                    feeddict[y_tf] = batch_y
                    opt, c = sess.run([optimiser, cost], feed_dict=feeddict)
                    avg_cost += c / n_batches
                    summary = sess.run(merged_summary, feed_dict=feeddict)
                    summary_writer.add_summary(summary, i)
                self.cost_list.append(avg_cost)


            self.all_weights = {}
            self.all_biases = {}

            for key, value in self.unique_ele.iteritems():
                w = []
                b = []
                for ii in range(len(all_weights[key])):
                    w.append(sess.run(all_weights[key][ii]))
                    b.append(sess.run(all_biases[key][ii]))
                self.all_weights[key]  = w
                self.all_biases[key] = b

    def predict(self, X):
        """
        This function uses the X data and plugs it into the model and then returns the predicted y

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :return: array of size (n_samples,)

            This contains the predictions for the target values corresponding to the samples contained in X.

        """

        # Splitting the X into the different descriptors
        X_input = self.__split_input(X)

        # Making a list of the unique elements and one of all the elements in order
        self.unique_ele, self.all_atoms = self.__unique_elements()

        # Create a list of tensorflow placeholders, one item per atom in the system
        inputs = []

        with tf.name_scope('input_x'):
            for ii in range(0, len(self.labels)):
                inputs.append(tf.placeholder(tf.float32, [None, self.labels[ii][1]]))

        # Zipping the placeholders and the labels, so it is easy to know which weights to call
        data = zip(self.all_atoms, inputs)

        # Making the weights into tf.variables
        all_weights = {}
        all_biases = {}

        for key, value in self.unique_ele.iteritems():
            w = []
            b = []
            for ii in range(len(self.all_weights[key])):
                w.append(tf.Variable(self.all_weights[key][ii]))
                b.append(tf.Variable(self.all_biases[key][ii]))
            all_weights[key] = w
            all_biases[key] = b

        # Evaluating the model
        with tf.name_scope("atom_nn"):
            all_atom_ene = []
            for ii in range(len(self.all_atoms)):
                atom_ene = self.__atom_energy(data[ii], all_weights, all_biases)
                all_atom_ene.append(atom_ene)

        # Summing the results to get the total energy
        with tf.name_scope("tot_ene"):
            model_tot = all_atom_ene[0]
            for ii in range(len(all_atom_ene) - 1):
                model_tot = tf.add(all_atom_ene[ii + 1], model_tot)

        # Initialising variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            feeddict = {i: d for i, d in zip(inputs, X_input)}
            pred = sess.run(model_tot, feed_dict=feeddict)
            predictions = np.reshape(pred, (pred.shape[0],))

        print pred
        return predictions

    def plot_cost(self):
        """
        This function plots the cost as a function of training iterations. It can only be called after the model has
        been trained.

        :return: None
        """
        try:
            self.cost_list
        except AttributeError:
            raise AttributeError("No values for the cost. Make sure that the model has been trained with the function "
                            "fit().")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.cost_list, label="Train set", color="b")
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Cost Value')
        ax.legend()
        # plt.yscale("log")
        plt.show()

    def correlationPlot(self, X, y):
        """
        This function plots a correlation plot of the values that are in the data set and the NN predictions. It expects
        the target values to be in Hartrees.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :ylim: tuple of shape (2,) containing doubles

            These are the limits of the y values for the plot.

        :xlim: tuple of shape (2,) containing doubles

            These are the limits of the x values for the plot.
        """
        y_pred = self.predict(X)
        df = pd.DataFrame()
        df['High level calculated energies (Ha)'] = y
        df['NN predicted energies (Ha)'] = y_pred
        lm = sns.lmplot('High level calculated energies (Ha)', 'NN predicted energies (Ha)', data=df,
                        scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
        # lm.set(ylim=ylim)
        # lm.set(xlim=xlim)
        plt.show()

    def __unique_elements(self):
        """
        This function takes the 'labels' parameter and extracts the unique elements. These are placed into a dictionary
        where the value is the number of features that each unique element has in the descriptor.

        :return: dictionary of size (n_unique_elements)
        """
        feat_dict = {}
        all_atoms = []

        for ii in range(0,len(self.labels)):
            feat_dict[self.labels[ii][0]] = self.labels[ii][1]
            all_atoms.append(self.labels[ii][0])

        return feat_dict, all_atoms

    def __generate_weights(self, n_input_layer):
        """
        This function generates the weights and the biases for each element-specific neural network. It does so by
        looking at the size of the hidden layers. The weights are initialised randomly.

        :n_input_layer: number of features in the descriptor for one atom - int
        :return: lists (of length n_hidden_layers + 1) of tensorflow variables
        """

        weights = []
        biases = []

        # Weights from input layer to first hidden layer
        weights.append(tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[0], n_input_layer], stddev=0.01), name='weight_in'))
        biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[0]]), name='bias_in'))

        # Weights from one hidden layer to the next
        for ii in range(len(self.hidden_layer_sizes)-1):
            weights.append(tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[ii+1], self.hidden_layer_sizes[ii]], stddev=0.01), name='weight_hidden'))
            biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[ii+1]]), name='bias_hidden'))

        # Weights from lat hidden layer to output layer
        weights.append(tf.Variable(tf.truncated_normal([1, self.hidden_layer_sizes[-1]], stddev=0.01), name='weight_out'))
        biases.append(tf.Variable(tf.zeros([1]), name='bias_out'))

        return weights, biases

    def __atom_energy(self, zip_data, all_weights, all_biases):
        """
        This function calculates the single atom energy with the single atom neural networks. It uses zip_data, which
        contains the tf placeholders for an atom and its corresponding atom label. all_weights/all_biases are all
        the weights/biases and their label.

        :zip_data: a two item list where the first item is the atom label (string) and the second is a tf.placeholder
        :all_weights: Dictionaries where the key is the atom label and the value is a list of weights (of length [n_hidden_layers+1,].
        :all_biases: Dictionaries where the key is the atom label and the value is a list of biases (of length [n_hidden_layers+1,].
        :return: tf.tensor containing the activation of the output layer.
        """
        label = zip_data[0]
        tf_input = zip_data[1]

        # Obtaining the index of the weights that correspond to the right atom
        z = tf.add(tf.matmul(tf_input, tf.transpose(all_weights[label][0])), all_biases[label][0])
        h = tf.nn.tanh(z)

        for ii in range(len(self.hidden_layer_sizes) - 1):
            z = tf.add(tf.matmul(h, tf.transpose(all_weights[label][ii + 1])), all_biases[label][ii + 1])
            h = tf.nn.tanh(z)

        z = tf.add(tf.matmul(h, tf.transpose(all_weights[label][-1])), all_biases[label][-1])

        return z

    def __split_input(self, X):
        """
        This function takes the data where the descriptor of all the atoms are concatenated into one line. It then splits
        it into n_atoms different data sets that will all be fed into a different mini-network.

        :X: numpy array of shape (n_samples, n_features_tot)
        :return: list of numpy array of shape (n_samples, n_features)
        """
        split_X = []

        counter = 0
        for ii in range(0, len(self.labels)):
            idx1 = counter
            idx2 = counter + self.labels[ii][1]
            split_X.append(X[:,idx1:idx2])
            counter = counter + self.labels[ii][1]

        return split_X

    def __reg_cost(self, nn_energy, qm_energy, all_weights):
        """
        This function calculates the cost function with L2 regularisation. It requires the energies predicted by the
        neural network and the energies calculated through quantum mechanics.

        :nn_energy: tf.Variable of shape [n_samples, 1]
        :qm_energy: tf.placeholder of shape [n_samples, 1]
        :all_weights: list of tuples. each tuple contains an atom label and a list of weights (of length [n_hidden_layers+1,].
        :return: tf.Variable of shape [1]
        """

        err = tf.subtract(qm_energy, nn_energy, name="error")
        cost = tf.nn.l2_loss(err, name="unreg_cost")   # scalar
        reg_l2 = tf.Variable(tf.zeros([1]), name="reg_term") # scalar

        for key, value in self.unique_ele.iteritems():
            for ii in range(len(self.hidden_layer_sizes) + 1):
                reg_l2 = tf.add(reg_l2, tf.nn.l2_loss(all_weights[key][ii]))


        reg_l2 = tf.scalar_mul(self.alpha, reg_l2)
        cost_reg = tf.add(cost, reg_l2, name="reg_cost")

        return cost_reg

    def checkBatchSize(self):
        """
        This function is called to check if the batch size has to take the default value or a user-set value.
        If it is a user set value, it checks whether it is a reasonable value.

        :return: int

            The default is 100 or to the total number of samples present if this is smaller than 100. Otherwise it is
            checked whether it is smaller than 1 or larger than the total number of samples.
        """
        if self.batch_size == 'auto':
            self.batch_size = min(100, self.n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > self.n_samples:
                print "Warning: Got `batch_size` less than 1 or larger than sample size. It is going to be clipped"
                self.batch_size = np.clip(self.batch_size, 1, self.n_samples)
            else:
                self.batch_size = self.batch_size

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels. It calculates the R^2 value. It is used during the
        training of the model.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :sample_weight: array of shape (n_samples,)

            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)

        :return: double
            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.
        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

    def scoreFull(self, X, y):
        """
        This scores the predictions more thouroughly than the function 'score'. It calculates the r2, the root mean
        square error, the mean absolute error and the largest positive/negative outliers. They are all in the units of
        the data passed.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :return:
        :r2: double

            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.

        :rmse: double

            This is the root mean square error

        :mae: double

            This is the mean absolute error

        :lpo: double

            This is the largest positive outlier.

        :lno: double

            This is the largest negative outlier.

        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        lpo, lno = self.largestOutliers(y, y_pred)

        return r2, rmse, mae, lpo, lno

    def largestOutliers(self, y_true, y_pred):
        """
        This function calculates the larges positive and negative outliers from the predictions of the neural net.

        :y_true: array of shape (n_samples,)

            This contains the target values for each sample.

        :y_pred: array of shape (n_samples,)

            This contains the neural network predictions of the target values for each sample.

        :return:

        :lpo: double

            This is the largest positive outlier.

        :lno: double

            This is the largest negative outlier.
        """
        diff = y_pred - y_true
        lpo = np.amax(diff)
        lno = - np.amin(diff)

        return lpo, lno

if __name__ == "__main__":


    def testMatrix1():
        X = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.8 ],
             [0.1, 0.0, 0.0, 0.9, 0.0, 0.0, -0.5, -0.5, -0.5, 0.1, 0.5, 0.5, 0.6, 0.5, 0.5,],
                [-0.1, 0.0, 0.0, 1.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.3, 1.5, 2.5, 0.5,]])
        y = np.array([4.0, 3.0, 1.0])
        return X, y

    def testMatrix2():
        from sklearn import preprocessing as preproc
        n_samples = 50
        X = np.zeros((n_samples, 2))
        for i in range(n_samples):
            X[i,0] = i
        y = range(0,n_samples)
        X = preproc.StandardScaler().fit_transform(X)
        return X, y

    X, y = testMatrix2()

    nn = BPNN(hidden_layer_sizes=(5,), labels=[('N', 1), ('C',1)], max_iter=400, alpha=0.0, learning_rate_init=0.01, batch_size=5)
    nn.fit(X, y)
    nn.plot_cost()
    nn.correlationPlot(X, y)
