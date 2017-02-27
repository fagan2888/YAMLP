import numpy as np
import coulomb_matrix
import extractEnergy
import tensorflow as tf
import matplotlib.pyplot as plt


fileX = open("X.csv", "r")
fileY = open("Y.csv", "r")

descriptor = coulomb_matrix.coulombMatrix(fileX)
energy = extractEnergy.getEnergy(fileY)

## ----------------- ** Structure of neural network ** ---------

n_hidden_layer = 25

# Parameters for training
learning_rate = 0.001
learning_iterations = 200
eps = tf.constant(0.12, dtype=tf.float32)

# Other important parameters
n_samples = energy.shape[0]
n_feat = descriptor.shape[1]

## ----------------- ** Initial set up of the NN ** ---------

X_train = tf.placeholder(tf.float32, descriptor.shape)
Y_train = tf.placeholder(tf.float32, energy.shape)

weights1 = tf.Variable(tf.random_normal([n_hidden_layer, n_feat])*2*eps)
bias1 = tf.Variable(tf.random_normal([n_samples, n_hidden_layer])*2*eps)
weights2 = tf.Variable(tf.random_normal([1, n_hidden_layer])*2*eps)
bias2 = tf.Variable(tf.random_normal([n_samples, 1])*2*eps)

a1 = tf.matmul(X_train, tf.transpose(weights1)) + bias1     # output of layer1, size = n_sample x n_hidden_layer (linear activation function)
model = tf.matmul(a1, tf.transpose(weights2)) + bias2       # output of last layer, size = n_samples x 1
cost = tf.reduce_mean(tf.nn.l2_loss((model-Y_train)))       # using the quadratic cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## -------------------- ** Initialising the graph and running it ** ---------------

init = tf.initialize_all_variables()
cost_array = []

with tf.Session() as sess:
    sess.run(init)
    for iter in range(learning_iterations):
        opt, c = sess.run([optimizer, cost], feed_dict={X_train: descriptor, Y_train: energy})
        cost_array.append(c)

y = np.array(cost_array)
plt.plot(y)
plt.show()

