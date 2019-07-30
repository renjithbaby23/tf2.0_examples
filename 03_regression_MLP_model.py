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

# Creating a simple model 
model = tf.keras.models.Sequential([tf.keras.layers.Dense(30, activation="selu", input_shape=X_train.shape[1:]),
                                   tf.keras.layers.Dense(10, activation="selu"),
                                   tf.keras.layers.Dense(1)
                                   ])

# Compiling using huber loss
model.compile(loss=tf.keras.losses.Huber(), optimizer="adam")

# training
history = model.fit(X_train, y_train, 
                    epochs=100,
                    validation_data=(X_valid, y_valid),
                    verbose=2, 
                    batch_size=128)

# evaluating the mdoel
mse_test = model.evaluate(X_test, y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = model.predict(X_new)
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))
