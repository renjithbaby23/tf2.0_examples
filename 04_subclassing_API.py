"""
Both functional and sequential APIs are declarative.
Hence the model is static layers of graph.

To create models involve loops, varying shapes, conditional branching,
and other dynamic behaviors, or simply if you prefer a more 
imperative programming style, the Subclassing API is for you

"""
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

# Creating a wide and deep model with two set of inputs using subclassing API
class WideAndDeepModel(tf.keras.models.Model):
    """
    Subclassing API is similar to the functional API;
    separate the creation of the layers in the constructor 
    from their usage in the call() method
    """
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs) # handles standard args (e.g., name)
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
model = WideAndDeepModel()


# Preprocessing the data to input_A - features 0 to 4 (total 5)
# and input_B - features 2 to 7 (total 6)
X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:5], X_test_B[:5]

# Compiling using huber loss
# Notice how two losses, their weights and two metrices are passed
model.compile(loss=[tf.keras.losses.Huber(), "mse"], 
              loss_weights=[0.5, 0.5],
              optimizer="adam", 
              metrics=["mse", "mse"])

# training 
# Notice how two labels are provided for both the heads
history = model.fit((X_train_A, X_train_B), (y_train, y_train), 
                    epochs=50,
                    validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)),
                    verbose=2, 
                    batch_size=128)

"""
This extra flexibility comes at a cost: your model’s architecture is hidden
within the call() method, so Keras cannot easily inspect it, it cannot save or clone it,
and when you call the summary() method, you only get a list of layers, without any
information on how they are connected to each other. Moreover, Keras cannot check
types and shapes ahead of time, and it is easier to make mistakes. So unless you really
need that extra flexibility, you should probably stick to the Sequential API or the
Functional API.
"""
model.summary()
# Plotting the model to file 
# Note that for subclass API, plotting the model doesn't really work
tf.keras.utils.plot_model(model, to_file='wide_ande_deep_model_subclass_API.png', show_shapes=True)

# evaluating the mdoel
# See how getting separate losses and combined loss
total_loss, main_loss, aux_loss, main_mse, aux_mse = model.evaluate((X_test_A, X_test_B), (y_test, y_test), verbose=2)
# x = model.evaluate((X_test_A, X_test_B), (y_test, y_test), verbose=2)
print("total_loss:{}, main_loss:{}, aux_loss:{}, main_mse:{}, aux_mse:{}".\
      format(total_loss, main_loss, aux_loss, main_mse, aux_mse))


X_new = X_test[:5] # pretend these are new instances
y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
print("y_pred_main: ", y_pred_main.T[0].round(2))
print("y_pred_aux: ", y_pred_aux.T[0].round(2))
print("y_test: ", y_test[:5].round(2))
