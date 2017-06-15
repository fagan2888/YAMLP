import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

### ------------ ** Generating dataset ** -------------

x = np.arange(-2.0,2.0,0.1)
y = x**2
x_data = np.reshape(x, (len(x), 1))
y_data = x_data**2

### ------------ ** Architecture of NN ** ---------

n_hidden_units = 2
n_samples = len(x)
n_input_units = 1

learning_rate = 0.01
learning_iterations = 5000
eps = 0.01

## ----------------- ** Constructing the model ** ---------

X_train = tf.placeholder(tf.float32, [None, n_input_units])
Y_train = tf.placeholder(tf.float32, [None, 1])

weights1 = tf.Variable(tf.random_normal([n_hidden_units, n_input_units]) * 2 * eps - eps)
bias1 = tf.Variable(tf.random_normal([n_hidden_units]) * 2 * eps - eps)
weights2 = tf.Variable(tf.random_normal([1, n_hidden_units]) * 2 * eps - eps)
bias2 = tf.Variable(tf.random_normal([1])*2*eps - eps)

a1 = tf.matmul(X_train, tf.transpose(weights1)) + bias1     # output of layer1, size = n_sample x n_hidden_units
a1 = tf.nn.sigmoid(a1)
model = tf.matmul(a1, tf.transpose(weights2)) + bias2       # output of last layer, size = n_samples x 1
cost = tf.reduce_mean(tf.nn.l2_loss((model-Y_train)))       # using the quadratic cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

### -------------------- ** Initialising the graph and running it ** ---------------

init = tf.initialize_all_variables()
cost_array = []

with tf.Session() as sess:
    sess.run(init)
    for iter in range(learning_iterations):
        opt, c = sess.run([optimizer, cost], feed_dict={X_train: x_data, Y_train: y_data})
        cost_array.append(c)

    # For the network with one layer, you can plot the output of the hidden layer
    out_hidden = sess.run(a1, feed_dict={X_train: x_data})

    neuron1 = out_hidden[:,0]

    if out_hidden.shape[1] == 2:
        neuron2 = out_hidden[:,1]
    elif out_hidden.shape[1] == 3:
        neuron2 = out_hidden[:, 1]
        neuron3 = out_hidden[:, 2]

    prediction = sess.run(model, feed_dict={X_train: x_data})


### -------------------- ** Plotting ** ---------------

theCost = np.array(cost_array)
plt.plot(theCost)
plt.show()

y_pred = np.reshape(prediction, (len(x)))

fig1 = plt.figure(figsize=(7,7))
ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(211)


ax1.scatter(x, y, marker="o", c="red", label = "actual data")
ax1.scatter(x, y_pred, marker="o", c="green", label = "predicted data")
ax1.legend(loc=4)

ax2.scatter(x, neuron1, marker="v", c="yellow", label = "hidden neuron 1")
if out_hidden.shape[1] == 2:
    ax2.scatter(x, neuron2, marker="v", c="green", label = "hidden neuron 2")
elif out_hidden.shape[1] == 3:
    ax2.scatter(x, neuron2, marker="v", c="green", label="hidden neuron 2")
    ax2.scatter(x, neuron3, marker="v", c="blue", label="hidden neuron 3")
ax2.legend()


plt.show()


