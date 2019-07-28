# regression MLP model
import tensorflow as tf
from sklearn import model_selection, preprocessing
from sklearn import datasets

# Using the sklearn california housing data
housing = datasets.fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = model_selection.train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train_full, y_train_full)

# Standard scaling the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Creating a wide and deep model with two set of inputs
# See the model plot generated to get the NN architecture
input_A = tf.keras.layers.Input(shape=[5])
input_B = tf.keras.layers.Input(shape=[6])
hidden1 = tf.keras.layers.Dense(30, activation="selu")(input_A)
hidden2 = tf.keras.layers.Dense(20, activation="selu")(hidden1)
concat = tf.keras.layers.concatenate([input_B, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.models.Model(inputs=[input_A, input_B], outputs=[output])

# Plotting the model to file 
tf.keras.utils.plot_model(model, to_file='wide_ande_deep_model1.png', show_shapes=True)

# Preprocessing the data to input_A - features 0 to 4 (total 5)
# and input_B - features 2 to 7 (total 6)
X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:5], X_test_B[:5]

# Compiling using huber loss
model.compile(loss=tf.keras.losses.Huber(), optimizer="adam", metrics=["mse"])

# training
history = model.fit((X_train_A, X_train_B), y_train, 
                    epochs=100,
                    validation_data=((X_valid_A, X_valid_B), y_valid),
                    verbose=2, 
                    batch_size=128)

# evaluating the mdoel
mse_test = model.evaluate((X_test_A, X_test_B), y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = model.predict((X_new_A, X_new_B))
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))


# Saving and restoring models
"""
Keras will save both the model’s architecture (including every layer’s hyperparameters)
 and the value of all the model parameters for every layer (e.g., connection
weights and biases), using the HDF5 format. It also saves the optimizer (including its
hyperparameters and any state it may have)

This will work when using the Sequential API or the Functional API, 
but unfortunately not when using Model subclassing. 
However, you can use save_weights() and load_weights() to at least save and restore 
the model parameters (but you will need to save and restore everything else yourself).

model.to_json('my_model_architecture.json') --> saves only the architecture of the model
model.save_weights('my_model_weights.h5') --> saves only the weights of the model
loaded_model = tf.keras.models.model_from_json(json_string)
loaded_model.load_weights('my_model_weights.h5') --> loads weights to the model

"""
model.save("results/wide_ande_deep_model.h5")

loaded_model = tf.keras.models.load_model("results/wide_ande_deep_model.h5")

# evaluating with the loaded mdoel
mse_test = loaded_model.evaluate((X_test_A, X_test_B), y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = loaded_model.predict((X_new_A, X_new_B))
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))
