"""
When the training lasts several hours on large datasets, 
you should not only save your model at the end of training, 
but also save checkpoints at regular intervals during training. 
To tell the fit() method to save checkpoints use callbacks.

The fit() method accepts a callbacks argument that lets you specify a list of objects
that Keras will call during training at the start and end of training, at the start and end
of each epoch and even before and after processing each batch. For example, 
the ModelCheckpoint callback saves checkpoints of your model at regular intervals during
training, by default at the end of each epoch (period=1).

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

# Model checkpoint callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("results/best_keras_model.h5", 
                                                   monitor='val_loss', 
                                                   verbose=0, 
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=5)
"""
If you use a validation set during training, you can set save_best_only=True 
when creating the ModelCheckpoint . In this case, it will only save your model 
when its performance on the validation set is the best so far. This way, 
you do not need to worry about training for too long and overfitting the training
set: simply restore the last model saved after training, and this will be the best model
on the validation set. It is the easiest way to implement early stopping.
"""

# Early stopping callback
"""
Another way to implement early stopping is to simply use the EarlyStopping call‐
back. It will interrupt training when it measures no progress on the validation set for
a number of epochs (defined by the patience argument), and it will optionally roll
back to the best model. You can combine both callbacks to both save checkpoints of
your model (in case your computer crashes), and actually interrupt training early
when there is no more progress (to avoid wasting time and resources):
"""
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     min_delta=0, 
                                                     patience=10, 
                                                     verbose=0, 
                                                     mode='auto', 
                                                     baseline=None, 
                                                     restore_best_weights=True)

# Custom callback
"""
you can implement on_train_begin() , on_train_end() ,
on_epoch_begin() , on_epoch_end() , on_batch_begin() and on_batch_end() .

Callbacks can also be used during evaluation and predictions, should you
ever need them (e.g., for debugging). In this case, you should implement
on_test_begin() , on_test_end() , on_test_batch_begin() , or on_test_batch_end() 
(called by evaluate() ), 

or on_predict_begin() , on_predict_end() , on_predict_batch_begin() , 
or on_predict_batch_end() (called by predict() ).
"""
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("val/train: {:.2f}\n".format(logs["val_loss"] / logs["loss"]))

custom_cb = PrintValTrainRatioCallback()

# training - notice the large epochs
history = model.fit((X_train_A, X_train_B), y_train, 
                    epochs=1000,
                    validation_data=((X_valid_A, X_valid_B), y_valid),
                    verbose=2, 
                    batch_size=64, 
                    callbacks=[checkpoint_cb, early_stopping_cb, custom_cb])

# evaluating the mdoel
mse_test = model.evaluate((X_test_A, X_test_B), y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = model.predict((X_new_A, X_new_B))
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))


loaded_model = tf.keras.models.load_model("results/best_keras_model.h5")

# evaluating with the loaded mdoel - see the difference in mse
mse_test = loaded_model.evaluate((X_test_A, X_test_B), y_test, verbose=2)
X_new = X_test[:5] # pretend these are new instances
y_pred = loaded_model.predict((X_new_A, X_new_B))
print("y_pred: ", y_pred.T[0].round(2))
print("y_test: ", y_test[:5].round(2))
