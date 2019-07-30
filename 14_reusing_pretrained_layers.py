"""
It is generally not a good idea to train a very large DNN from scratch: instead, you should always 
try to find an existing neural network that accomplishes a similar task to the one you are trying 
to tackle, then just reuse the lower layers of this network - Transfer Learning.

Transfer learning will not only speed up training considerably, 
but will also require much less training data.

Try freezing all the reused layers first (i.e., make their weights non-trainable, so gradient descent 
won’t modify them), then train your model and see how it performs. Then try unfreezing one or two of 
the top hidden layers to let backpropagation tweak them and see if performance improves. 
The more training data you have, the more slayers you can unfreeze. It is also useful to reduce 
the learning rate when you unfreeze reused layers: this will avoid wrecking their fine-tuned weights.
If you still cannot get good performance, and you have little training data, try dropping the top 
hidden layer(s) and freeze all remaining hidden layers again. You can iterate until you find the 
right number of layers to reuse. If you have plenty of training data, you may try replacing the 
top hidden layers instead of dropping them, and even add more hidden layers.

"""

import tensorflow as tf

# loading an existing model for task A and using it to solve task B
model_A = tf.keras.models.load_model("models/my_model_A.h5")

model_B_on_A = tf.keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Note that model_A and model_B_on_A now share some layers. When you train model_B_on_A , 
# it will also affect model_A. If you want to avoid that, you need to clone model_A 
# before you reuse its layers.
model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

# freezing the reused layere
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# Always compile your model after you freeze or unfreeze layers
model_B_on_A.compile(loss="binary_crossentropy", 
                     optimizer="sgd",
                     metrics=["accuracy"])

# we can train the model for a few epochs, then unfreeze the reused layers 
# (which requires compiling the model again) and continue training to fine-tune the reused
# layers for task B. After unfreezing the reused layers, it is usually a good idea to reduce
# the learning rate, once again to avoid damaging the reused weights:
X_train_B = None
y_train_B = None
X_valid_B = None
y_valid_B = None

history = model_B_on_A.fit(X_train_B, y_train_B, 
                           epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

# making the layers trainable again
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

# compiling again with a reduced learning rate
optimizer = tf.keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", 
                     optimizer=optimizer,
                     metrics=["accuracy"])

# training again on task B
history = model_B_on_A.fit(X_train_B, y_train_B, 
                           epochs=16,
                           validation_data=(X_valid_B, y_valid_B))


# It turns out that transfer learning does not work very well with small dense networks: 
# it works best with deep convolutional neural networks


######## Unsupervised pretraining ###########

"""
Suppose you want to tackle a complex task for which you don’t have much labeled training data,
but you couldn't find a model trained on a similar problem, you can think of unsupervised pretraining.

If you can gather plenty of unlabeled training data, you can try to train the layers one by one, 
starting with the lowest layer and then going up, using an unsupervised feature detector algorithm 
such as Restricted Boltzmann Machines autoencoders.
* Each layer is trained on the output of the previously trained layers 
* All layers except the one being trained are frozen
* Once all layers have been trained this way, you can add the output layer for your task, 
    and fine-tune the final network using supervised learning

Until 2010, unsupervised pretraining (typically using RBMs) was the norm for deep nets, 
and it was only after the vanishing gradients problem was alleviated that it became much 
more common to train DNNs purely using supervised learning. 
However, unsupervised pretraining (today typically using autoencoders rather than RBMs) 
is still a good option when you have a complex task to solve, no similar model you can reuse, 
and little labeled training data but plenty of unlabeled training data.
"""

###### Pretraining on auxilary task ########
"""
If you do not have much labeled training data, one last option is to train a first neural
network on an auxiliary task for which you can easily obtain or generate labeled
training data, then reuse the lower layers of that network for your actual task. The
first neural network’s lower layers will learn feature detectors that will likely be reusa‐
ble by the second neural network.
"""
