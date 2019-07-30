"""
To use tensorboard, you must modify your program so that it outputs the data you want 
to visualize to special binary log files called event files. Each binary data record is called a
summary. The TensorBoard server will monitor the log directory, and it will automatically 
pick up the changes and update the visualizations: this allows you to visualize
live data (with a short delay), such as the learning curves during training. 
In general, you want to point the TensorBoard server to a root log directory, and configure your
program so that it writes to a different subdirectory every time it runs. This way, the
same TensorBoard server instance will allow you to visualize and compare data from
multiple runs of your program, without getting everything mixed up.
"""
import tensorflow as tf
from sklearn import model_selection, preprocessing
from sklearn import datasets
import os

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

# Preprocessing the data to input_A - features 0 to 4 (total 5)
# and input_B - features 2 to 7 (total 6)
X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:5], X_test_B[:5]

# Compiling using huber loss
model.compile(loss=tf.keras.losses.Huber(), optimizer="adam", metrics=["mse"])

# Early stopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     min_delta=0, 
                                                     patience=10, 
                                                     verbose=0, 
                                                     mode='auto', 
                                                     baseline=None, 
                                                     restore_best_weights=True)

# Tensorboard logdir creation
root_logdir = os.path.join(os.curdir, "logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
print("loggin to: ", run_logdir)

# Tensorboard callback
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=run_logdir, 
                                                histogram_freq=0, 
                                                batch_size=32, 
                                                write_graph=True, 
                                                write_grads=False, 
                                                write_images=False, 
                                                embeddings_freq=0, 
                                                embeddings_layer_names=None, 
                                                embeddings_metadata=None, 
                                                embeddings_data=None, 
                                                update_freq='epoch')

# training - notice the large epochs
history = model.fit((X_train_A, X_train_B), y_train, 
                    epochs=1000,
                    validation_data=((X_valid_A, X_valid_B), y_valid),
                    verbose=2, 
                    batch_size=64, 
                    callbacks=[early_stopping_cb, tensorboard_cb])

# evaluating the mdoel
mse_test = model.evaluate((X_test_A, X_test_B), y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = model.predict((X_new_A, X_new_B))
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))

