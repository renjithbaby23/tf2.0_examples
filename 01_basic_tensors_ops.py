import tensorflow as tf
import numpy as np
import time

print(tf.__version__)

x = tf.add(1, 2)
# print(x.numpy())

a = np.array([1, 2, 3, 4])
b = np.array([1, 0, 1, 0])

y = tf.add(a, b)
# print(y.numpy())

# print(tf.reduce_sum(y))

# print(tf.square(y))

x = tf.matmul([[2]], [[2, 3]])
# print(x)

y = tf.multiply([[2, 3], [3,4]], [[2, 3], [3,4]])

# print(y)
# print(y.shape)
# print(y.dtype)
# print(y.numpy())

x = tf.random.uniform([3, 3])

# print("Is there a GPU available: "),
# print(tf.test.is_gpu_available())

# print("Is the Tensor on GPU #0:  "),
# print(x.device.endswith('GPU:0'))

# print("Is the Tensor on RAM #0:  "),
# print(x.device.endswith('CPU:0'))


def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)