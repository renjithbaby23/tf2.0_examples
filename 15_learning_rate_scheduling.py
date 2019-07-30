"""
Sample implementations of learning rate scheduling
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

print("class_names[y_train[0]] :", class_names[y_train[0]])

# Defining the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(300, activation=tf.keras.activations.selu))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.selu))
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

# printing the summary
model.summary()

##########################################
# Learning rate scheduling methods

# 1. Power scheduling - implement by providing the decay values
optimizer1 = tf.keras.optimizers.SGD(lr=0.01, decay=1e-4)

# 2. Exponential scheduling - implement the below functions
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn
    
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
# then define a callback and use it while training
lr_scheduler_cb1 = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# 3. Performance scheduling - use the below callback
lr_scheduler_cb2 = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

# Using the tf.keras.optimizers api
# an alternative way to implement learning rate scheduling: just define the learning rate 
# using one of the schedules available in keras.optimizers.schedules , then pass this 
# learning rate to any optimizer. This approach updates the learning rate at each step 
# rather than at each epoch. 

# a sample implementation of exponential schedule as earlier:

s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer2 = tf.keras.optimizers.SGD(learning_rate)

##########################################

# Compiling the model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=optimizer2, 
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

# training
history = model.fit(X_train, y_train,
                    epochs=10, 
                    validation_data=(X_valid, y_valid), 
                    batch_size=64, 
                    verbose=2)

# Evaluating on test data
print("Evaluation result: ")
model.evaluate(X_test, y_test)

# Using the model to make predictions
X_sample = X_test[:5]
y_proba = model.predict(X_sample)
print(y_proba.round(2))

y_pred = model.predict_classes(X_sample)
print("y_pred: ", y_pred)
print("y_test: ", y_test[:5])
print("predicted classes: ", np.array(class_names)[y_pred])
print("actual classes: ", np.array(class_names)[y_test[:5]])
