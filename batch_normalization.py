"""
Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce 
the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee 
that they won’t come back during training.

Batch Normalization (BN):
It consists of adding an operation in the model just before or after the activation function of 
each hidden layer, simply zero-centering and normalizing each input, then scaling and shifting 
the result using two new parameter vectors per layer: one for scaling, the other for shifting. 

In many cases, if you add a BN layer as the very first layer of your neural network, you do not need to
standardize your training set: the BN layer will do it for you (well, approximately, since it only 
looks at one batch at a time, and it can also rescale and shift each input feature).

In order to zero-center and normalize the inputs, the algorithm needs to estimate each input’s mean and 
standard deviation. It does so by evaluating the mean and standard deviation of each input over 
the current mini-batch (hence the name “Batch Normalization”).

During training, the batches should not be too small, if possible more than 30 instances, 
and all instances should be independent and identically distributed (IID).

############# What happens during testing? ###############
So during training, BN just standardizes its inputs then rescales and offsets them.
Good! What about at test time? Well it is not that simple. Indeed, we may need to
make predictions for individual instances rather than for batches of instances: in this
case, we will have no way to compute each input’s mean and standard deviation.

four parameter vectors are learned in each batch-normalized layer: 
    γ (the ouput scale vector) β (the output offset vector) 
                                                - learned through regular backpropagation
    μ (the final input mean vector), and σ (the final input standard deviation vector) 
                                                - estimated using an exponential moving average. 

    ############# Equations guiding a batchnorm layer###############
    μ_B = mean of batch input (of all x_i)
    σ_B = standard deviation of batch input (of all x_i)
    x_hat_i = (x_i - μ_B)/σ_B
    z_i = hadmad_product(γ, x_hat_i) + β    ,which is the output of the batchnorm layer

Note that μ and σ are estimated during training, but they are not used at all during training, 
only after training (to replace the batch input means and standard deviations

##### Observations of the authors after using batchnorm ######
    1. The vanishsing gradients problem was strongly reduced, to the point that 
        they could use saturating activation functions such as the tanh and even the logistic activation function.
    2. The networks were also much less sensitive to the weight initialization. 
    3. Were able to use much larger learning rates, significantly speeding up the learning process.
    4. It also acts like a regularizer, reducing the need for other regularization techniques.
    5. Increases the model complexity and there is a runtime penalty: the neural network makes slower predictions 
        due to the extra computations required at each layer.
        ** So if you need predictions to be lightning-fast, 
        you may want to check how well plain ELU + He initialization 
        perform before playing with Batch Normalization.**

"""

import tensorflow.keras as keras

# Model with BN after activation
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                keras.layers.BatchNormalization(), # optional first batchnorm layer
                                keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dense(10, activation="softmax")
                                ])

# Note that the input shape equals to output shape for batchnorm layers, 
# and trainable parameters is four times the number of input dimension

model.summary()

# Printing the parameter name and trainable? of first batchnorm layer
print([(var.name, var.trainable) for var in model.layers[1].variables])

# Notice that two are trainable (by backprop), and two are not.
# As mentioned earlier, the remaining two are updated using moving average during from batches


########### How the moving_mean and moving_variance are updated during training? #########
print(model.layers[1].updates)
# Now when you create a BN layer in Keras, it also creates two operations that will be
# called by Keras at each iteration during training. These operations will update the
# moving averages. These operations are TensorFlow operations.


# Model with BN before activation
# To add the BN layers before the activation functions, we must remove the 
# activation function from the hidden layers, and add them as separate layers 
# after the BN layers. Since a Batch Normalization layer includes one offset parameter 
# per input (β - beta), you can remove the bias term from the previous layer 
# (just pass use_bias=False when creating it)
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
                                keras.layers.BatchNormalization(),
                                keras.layers.Activation("elu"),
                                keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
                                keras.layers.Activation("elu"),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dense(10, activation="softmax")
                                ])

model.summary()

# Momentum hyperparameter of batchnorm layer - is used when updating the exponential moving averages: 
# given a new value v, the moving average v_hat is updated as follows.
# v_hat <-- v_hat × momentum + v × (1 − momentum)
# A good momentum value is typically close to 1—for example, 0.9, 0.99, or 0.999 (you
# want more 9s for larger datasets and smaller mini-batches)

##################################################################
# As BN is tricky to use with RNNs, gradient clipping is generally used to prevent exploding gradients
# For other types of networks, BN is usually sufficient.
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)

# This will clip every component of the gradient vector to a value between –1.0 and 1.0.
# This means that all the partial derivatives of the loss (with regards to each and every
# trainable parameter) will be clipped between –1.0 and 1.0. The threshold is a hyper‐
# parameter you can tune.

"""
If you want to ensure that Gradient Clipping does not change
the direction of the gradient vector, you should clip by norm by setting clipnorm
instead of clipvalue . This will clip the whole gradient if its l 2 norm is greater than
the threshold you picked. For example, if you set clipnorm=1.0 , then the vector [0.9,
100.0] will be clipped to [0.00899964, 0.9999595], preserving its orientation, but
almost eliminating the first component. If you observe that the gradients explode
during training (you can track the size of the gradients using TensorBoard), you may
want to try both clipping by value and clipping by norm, with different threshold,
and see which option performs best on the validation set.
"""


