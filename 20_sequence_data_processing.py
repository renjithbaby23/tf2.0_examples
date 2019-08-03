"""
Preparing a sample time series dataset and preprocessing it to be ready to be used by neural networks.

Requires tensorflow 2.x
"""
import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.range(10)

# windowing the dataset with shift
# try without drop_remainder
dataset = dataset.window(5, shift=1, drop_remainder=True)
print("dataset = dataset.window(5, shift=1, drop_remainder=True) : ")
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print("")

# Converting the dataset to a numpy array
# Use flat_map if you want to make sure that the order of your dataset stays the same.
dataset = dataset.flat_map(lambda window: window.batch(5))
print("dataset = dataset.flat_map() : ")
for i in dataset:
    print(i.numpy())

# making the x and y from the numpy arrays
# map will execute one function on every element of the Dataset separately, 
# whereas apply will execute one function on the whole Dataset at once 
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

# buffer_size -> amount of data items we have in the dataset
# pickes the first buffer_size number of items and shuffles and then pickes 
# the next buffer_size elements, asd so on
"""
Shuffling to avoid sequence bias.
Sequence bias is when the order of things can impact the selection of things. 
For example, if I were to ask you your favorite TV show, and listed 
"Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, 
you're probably more likely to select 'Game of Thrones' as you are familiar with it, 
and it's the first thing you see. Even if it is equal to the other TV shows. 
So, when training data in a dataset, we don't want the sequence to impact the training 
in a similar way, so it's good to shuffle them up.
"""
dataset = dataset.shuffle(buffer_size=10)
print("dataset = datase.map(lambda window: (window[:-1], window[-1:]))  : ")
for x, y in dataset:
    print(x.numpy(), y.numpy())

# batching the data

dataset = dataset.batch(2).prefetch(1)
print("dataset = dataset.batch(2).prefetch(1) : ")
for x, y in dataset:
    print("x: ", x.numpy())
    print("y: ", y.numpy())

# Creating a function to do the preprocessing on a series of data

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# feeding the sample data to a single layer linear regression NN

# creating a dummy dataset
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)


# spliting the dataset into train and val set
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_val = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

layer_0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([layer_0])

print("training the model...")
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)

print("layer weights: {}".format(layer_0.get_weights()))

print("series[1:21]: ", series[1:21])
temp_result = model.predict(series[1:21][np.newaxis])
print("temp_result: ", temp_result)

# forecasting for the entire series
forecast= []

for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
result = np.array(forecast)[:, 0, 0]

error = tf.keras.metrics.mean_absolute_error(x_val, result).numpy() 
print("error : ", error)
