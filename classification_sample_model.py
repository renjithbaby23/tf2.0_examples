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
print("model.layers: ", model.layers)

# Layer name from layer
print("model.layers[2].name: ", model.layers[2].name)

#Getting a layer info from layer name
print("model.get_layer('dense_1').name: ", model.get_layer('dense_1').name)
print("model.get_layer('dense_1'): ", vars(model.get_layer('dense_1')).keys())

hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
print("weights.shape: ", weights.shape)
print("biases.shape: ", biases.shape)
# Printing the dense_1 layer weights and biases
# Note how the weights are randomly initialized to break symmetry
# Whereas the biases are initialized to zeros, which is fine 
print("model.get_layer('dense_1').trainable_weights: ", model.get_layer(model.layers[1].name).trainable_weights)

# Compiling the model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, # Sparse because our labels are not one hot encoded
              optimizer=tf.keras.optimizers.SGD(), # simple stochastic gradient descent
              metrics=[tf.keras.metrics.sparse_categorical_accuracy]) # since we used sparse_categorical_crossentropy loss
# metrics=["accuracy"] is equivalent to 
# metrics=[keras.metrics.sparse_categorical_accuracy] (when using this loss)

# # Sample of one hot encoder conversion of label
# # printing the first train label
# print(y_train[0])
# # a-> one hot encoder vector of first label
# a = tf.keras.utils.to_categorical(y_train[0], num_classes=10)
# print(a)
# # reversing the conversion
# import numpy as np
# print(np.argmax(a))

# training
history = model.fit(X_train, y_train,
                    epochs=10, 
                    validation_data=(X_valid, y_valid), 
                    batch_size=64, 
                    verbose=2)

# ploting the train and validation loss and accuracy

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()

# Evaluating on test data
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
