"""
Implement XOR gate using neural networks
Use tensorflow 2.x
"""

import tensorflow as tf
import numpy as np

# # Generating a dummy data - XOR Gate

x = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

# Creating the model
# Methode 1 - implementing the Model class
# Method 2 - tf.keras Sequential or functional APIs can also be used

#implementing the Model class

class xor_net(tf.keras.Model):
    def __init__(self):
        super(xor_net, self).__init__()
        self.input1 = tf.keras.layers.Dense(2, activation='sigmoid', input_dim=2)
        self.output1 = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, input):
        x = self.input1(input)
        # x = self.input2(x)
        return self.output1(x)
model = xor_net()

# Try changing the lr to 0.1, 0.01, 0.001 and 0.001 
# and observe the loss to see the variation in speed of learning
optimizer = tf.keras.optimizers.Adam(lr=0.01)
epochs = 2000

# Implementing gradient descent
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = tf.keras.losses.binary_crossentropy(y, prediction)
        loss = tf.reduce_sum(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 50 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss))

# result
print("result: \n", np.round(model(x).numpy()))

print("weights - hidden layer: ", model.weights[0].numpy())
print("weights - output layer: ", model.weights[1].numpy())
