"""
Reference: https://www.tensorflow.org/beta/guide/effective_tf2

Implement just an alternate - more powerful way to do 
gradient descent using tf.GradientTape()

"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
print(tf.__version__)

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

y_train = K.one_hot(y_train, num_classes=10)
y_test = K.one_hot(y_test, num_classes=10)

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


for layer in model.layers:
    print(layer.name)
    print("input_shape: ", layer.input_shape)
    print("output_shape: ", layer.output_shape)

for x, y in zip(x_train, y_train):
    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)
    # print("x.shape: ", x.shape)
    # print("y.shape: ", y)
    with tf.GradientTape() as tape:
        prediction = model(x)
        # print("prediction: ",prediction)
        loss = tf.nn.softmax_cross_entropy_with_logits(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        adam = tf.keras.optimizers.Adam()
        adam.apply_gradients(zip(gradients, model.trainable_variables))

count = 0

while count < 10:
    x, y = x_train[count], y_train[count]
    x = np.expand_dims(x, axis=0)
    print("*"*20)

    prediction = model(x)
    print("Predicted: %i, Actual: %i " %(np.argmax(prediction), np.argmax(y)))

    print("*"*20)
    count += 1

