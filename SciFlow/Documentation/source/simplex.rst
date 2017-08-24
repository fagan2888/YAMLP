Example 1: Cubic function
***************************

This example shows how to use the estimator MLPRegFlow to fit a cubic function. The following modules and packages need importing::

    import numpy as np
    import NNFlow as nn
    import matplotlib.pyplot as plt

The first step is to generate the data set, which here is a cubic function::

    x = np.arange(-2.0, 2.0, 0.05)
    X = np.reshape(x, (len(x), 1))
    y = np.reshape(X ** 3, (len(x),))

Then, the estimator is defined::

    estimator = nn.MLPRegFlow(hidden_layer_sizes=(5,), learning_rate_init=0.01, max_iter=5000, alpha=0)
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

The second line fits the model to the data and then the third line predicts the values of the same data. Here no training/test set separation is used, as it is only a test to check whether the model is working. Once this is done, a correlation plot and a plot of the predictions can be generated::

    #  Visualisation of predictions
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(x, y, label="original", marker="o", c="r")
    ax2.scatter(x, y_pred, label="predictions", marker="o", c='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()

    # Correlation plot
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.scatter(y, y_pred, marker="o", c="r")
    ax3.set_xlabel('original y')
    ax3.set_ylabel('prediction y')
    plt.show()

These should look like this:

.. image:: Test1-1.png
    :width: 600px
    :height: 600px
    :scale: 50 %
    :alt: alternate text
    :align: left

.. image:: Test1-2.png
    :width: 600px
    :height: 600px
    :scale: 50 %
    :alt: alternate text
    :align: right
