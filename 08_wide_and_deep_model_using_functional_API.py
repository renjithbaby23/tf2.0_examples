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
