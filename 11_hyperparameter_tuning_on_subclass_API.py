"""
Fine tuning hyperparameters using tf.keras sklearn wrapper
This example uses model created using subclass API

"""
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
from sklearn import datasets
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


# Using the sklearn california housing data
housing = datasets.fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = model_selection.train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train_full, y_train_full)
X_new = X_test[:5] # pretend these are new instances for prediction

# Standard scaling the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Creating a wide and deep model with two set of inputs using subclassing API
class subclassModel(tf.keras.models.Model):
    """
    Creating the model
    """
    def __init__(self, n_neurons=30, activation="relu"):
        super().__init__() # handles standard args (e.g., name)
        self.hidden1 = tf.keras.layers.Dense(n_neurons, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(n_neurons, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
    def call(self, inputs):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        main_output = self.main_output(hidden2)
        return main_output

def build_model(n_neurons=30, learning_rate=3e-3):
    model = subclassModel(n_neurons=n_neurons)

    # Defining the optimizer and compiling
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Using sklearn wrapper
keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)

# defining the parameter dictionary to define the hyper parameter space
param_distribs = {"n_neurons": np.arange(20, 100),
                  "learning_rate": reciprocal(3e-3, 3e-1),
                    }

# Random grid search with 3 fold cross validation
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, 
                  epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)], 
                  verbose=2)

print("rnd_search_cv.best_params_: ", rnd_search_cv.best_params_)
print("rnd_search_cv.best_score_: ", rnd_search_cv.best_score_)

# Getting the best model and doing the inference
model = rnd_search_cv.best_estimator_.model
mse_test = model.evaluate(X_test, y_test)
print("mse_test: ", mse_test)
y_pred = model.predict(X_new)
