"""
Regularization:

With so many parameters, the network has an incredible amount of freedom and can fit a huge variety 
of complex datasets. But this great flexibility also means that it is prone to overfitting the training set. 
We need regularization. 

Early stopping is one of the best regularization we have seen. 
Moreover, even though Batch Normalization was designed to solve the vanishing/exploding gradients problems, 
is also acts like a pretty good regularizer.

We will see other popular regularization techniques for neural networks: 
    l1 and l2 regularization
    Dropout regularization
    Monte Carlo Dropout regularization
    max-norm regularization
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("X_train_full.shape: ", X_train_full.shape)
print("y_train_full.shape: ", y_train_full.shape)
print("X_test.shape: ", X_test.shape)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", \
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 1. model with l1 and l2 regularization
"""
for l1 -> kernel_regularizer=tf.keras.regularizers.l1(0.01), add this argument while creating layers
for l2 -> kernel_regularizer=tf.keras.regularizers.l2(0.01)
for l1 and l2 -> kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.02)
eg:
    layer = tf.keras.layers.Dense(100, activation="elu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))

* When you want to apply same regularization to multiple layers, 
better use python’s functools.partial() function

"""

# Defining the model
from functools import partial
RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=[28, 28]),
                                RegularizedDense(300),
                                RegularizedDense(100),
                                RegularizedDense(10, activation="softmax",
                                                 kernel_initializer="glorot_uniform")
                                ])

# printing the summary
model.summary()

# 2. model with Dropout regularization
"""
At every training step, every neuron (including the input neurons, but always excluding the output neurons) 
has a probability p of being temporarily “dropped out,” meaning it will be entirely ignored during this 
trainingstep, but it may be active during the next step. The hyperparameter 'p' is called the dropout rate, 
and it is typically set to 50%. After training, neurons don’t get dropped anymore. 
"""

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=[28, 28]),
                                   tf.keras.layers.Dropout(rate=0.2),
                                   tf.keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
                                   tf.keras.layers.Dropout(rate=0.2),
                                   tf.keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
                                   tf.keras.layers.Dropout(rate=0.2),
                                   tf.keras.layers.Dense(10, activation="softmax")
                                   ])

"""
Since dropout is only active during training, the training loss is penalized compared to the validation loss, 
so comparing the two can be misleading. In particular, a model may be overfitting the training set and yet 
have similar training and validation losses. So make sure to evaluate the training loss without dropout 
(e.g., after training). Alternatively, you can call the fit() method inside a with 
keras.backend.learning_phase_scope(1) block: this will force dropout to be active during both training and validation.
"""

# Compiling the model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, 
              optimizer=tf.keras.optimizers.SGD(), 
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

# training
history = model.fit(X_train, y_train,
                    epochs=10, 
                    validation_data=(X_valid, y_valid), 
                    batch_size=64, 
                    verbose=2)

# 3. Monte Carlo dropout
"""
MC Dropout, can boost the performance of any trained dropout model, without having to retrain it
or even modify it at all! We first force training mode on, using a learning_phase_scope(1) context. 
This turns dropout on within the with block. Then we make 100 predictions over the test set, 
and we stack them. Since dropout is on, all predictions will be different. Recall that predict() 
returns a matrix with one row per instance, and one column per class. Since there are 10,000 instances 
in the test set, and 10 classes, this is a matrix of shape [10000, 10]. We stack 100 such matrices, 
so y_probas is an array of shape [100, 10000, 10]. Once we average over the first dimension 
( axis=0 ), we get y_proba , an array of shape [10000, 10], like we would get with a single prediction. 
That’s all!

Averaging over multiple predictions with dropout on gives us a Monte Carlo estimate that is
generally more reliable than the result of a single prediction with dropout off.

The number of Monte Carlo samples you use (100 in this example) is a hyperparameter you can tweak. 
The higher it is, the more accurate the predictions and their uncertainty estimates will be. 
However, it you double it, inference time will also be doubled.

If your model contains other layers that behave in a special way during training (such as Batch 
Normalization layers), then you should not force training mode like we just did. Instead, you should 
replace the Dropout layers with the following MCDropout class:

    class MCDropout(keras.layers.Dropout):
        def call(self, inputs):
            return super().call(inputs, training=True)

We just sublass the Dropout layer and override the call() method to force its training argument to True.

"""
with tf.keras.backend.learning_phase_scope(1): # force training mode => dropout on
    y_probas = np.stack([model.predict(X_test) for _ in range(100)])

y_pred = y_probas.mean(axis=0)

# 4. MaxNorm regularization
"""
max-norm regularization in tf.keras, just set every hidden layer’s kernel_constraint argument 
to a max_norm() constraint, with the appropriate maxvalue, 
for example:
    tf.keras.layers.Dense(100, activation="elu", 
                            kernel_initializer="he_normal",
                            kernel_constraint=tf.keras.constraints.max_norm(1.))

After each training iteration, the model’s fit() method will call the object returned by max_norm(), 
passing it the layer’s weights and getting clipped weights in return, which then replace the layer’s weights.
"""


X_sample = X_test[:5]
y_pred = model.predict_classes(X_sample)
print("y_pred: ", y_pred)
print("y_test: ", y_test[:5])
print("predicted classes: ", np.array(class_names)[y_pred])
print("actual classes: ", np.array(class_names)[y_test[:5]])

