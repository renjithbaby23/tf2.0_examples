"""
* This script just lists GENERALLY BEST activation functions and
the best kernel initializers associated with them.

Glorot and Bengio propose a way to significantly alleviate the problem with vanishing gradients.
We need the signal to flow properly in both directions: in the forward direction when making predictions, 
and in the reverse direction when backpropagating gradients. 
We don’t want the signal to die out, nor do we want it to explode and saturate.
For the signal to flow properly, we need the variance of the
outputs of each layer to be equal to the variance of its inputs, and we also need the
gradients to have equal variance before and after flowing through a layer in the
reverse direction. It is actually not possible to guarantee both unless the layer has an equal
number of inputs and neurons (these numbers are called the fan_in and fan_out of the
layer), but they proposed a good compromise that has proven to work very well in
practice: the connection weights of each layer must be initialized as a uniform on normal di
sribution with specific mean and variance.

fan_avg = fan_in + fan_out /2

|Initialization | Activation functions               | σ2 (Normal)|
|-----------------------------------------------------------------|
|Glorot         | None, Tanh, Logistic, Softmax      | 1 / fan_avg|
|He             | ReLU & variants                    | 2 / fan_in |
|LeCun          | SELU                               | 1 / fan_in |
 -----------------------------------------------------------------


###############################################################
So which activation function should you use for the hidden layers of your 
deep neural networks? Although your mileage will vary, in general 
SELU > ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic. 
If the network’s architecture prevents it from self-normalizing, 
then ELU may perform better than SELU (since SELUis not smooth at z = 0). 
If you care a lot about runtime latency, then you may prefer leaky ReLU. 
If you don’t want to tweak yet another hyperparameter, you may just use 
the default α values used by Keras (e.g., 0.3 for the leaky ReLU). 
If you have spare time and computing power, you can use cross-validation 
to evaluate other activation functions, in particular RReLU if your network 
is over‐fitting, or PReLU if you have a huge training set.
###############################################################
"""
import tensorflow as tf

# ReLU -> Best use with He Normal initialization
tf.keras.layers.Dense(10, 
                      activation="relu", 
                      kernel_initializer="he_normal")

# Leaky ReLU -> Best use with He Normal initialization
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
layer = tf.keras.layers.Dense(10, 
                              activation=leaky_relu, 
                              kernel_initializer="he_normal")

# SELU -> Best use with LeCunn Normal initialization
layer = tf.keras.layers.Dense(10, 
                              activation="selu", 
                              kernel_initializer="lecun_normal")

# If you want He initialization with a uniform distribution, but based on fan avg rather
# than fan_in , you can use the VarianceScaling initializer like this:

he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., 
                                                    mode='fan_avg', 
                                                    distribution='uniform')
tf.keras.layers.Dense(10, 
                      activation="sigmoid", 
                      kernel_initializer=he_avg_init)


# Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce 
# the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee 
# that they won’t come back during training. That's where batch normalization comes in to picture.
