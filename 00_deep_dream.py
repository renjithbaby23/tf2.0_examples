import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

# Download an image and read it into a NumPy array, 
def download(url):
  name = url.split("/")[-1]
  image_path = tf.keras.utils.get_file(name, origin=url)
  img = image.load_img(image_path)
  return image.img_to_array(img)

# Scale pixels to between (-1.0 and 1.0)
def preprocess(img):
  return (img / 127.5) - 1
  
# Undo the preprocessing above
def deprocess(img):
  img = img.copy()
  img /= 2.
  img += 0.5
  img *= 255.
  return np.clip(img, 0, 255).astype('uint8')

# Display an image
def show(img):
  plt.figure(figsize=(12,12))
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img)

# https://commons.wikimedia.org/wiki/File:Flickr_-_Nicholas_T_-_Big_Sky_(1).jpg
url = 'https://storage.googleapis.com/applied-dl/clouds.jpg'
img = preprocess(download(url))
show(deprocess(img))


inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet',
                                                 include_top=False)

# We'll maximize the activations of these layers
names = ['mixed2', 'mixed3', 'mixed4', 'mixed5']
layers = [inception_v3.get_layer(name).output for name in names]

# Create our feature extraction model
feat_extraction_model = tf.keras.Model(inputs=inception_v3.input, outputs=layers)

def forward(img):
  
  # Create a batch
  img_batch = tf.expand_dims(img, axis=0)
  
  # Forward the image through Inception, extract activations
  # for the layers we selected above
  return feat_extraction_model(img_batch)

def calc_loss(layer_activations):
  
  total_loss = 0
  
  for act in layer_activations:
    
    # In gradient ascent, we'll want to maximize this value
    # so our image increasingly "excites" the layer
    loss = tf.math.reduce_mean(act)

    # Normalize by the number of units in the layer
    loss /= np.prod(act.shape)
    total_loss += loss

  return total_loss

# Convert our image into a variable for training
img = tf.Variable(img)

# Run a few iterations of gradient ascent
steps = 100

for step in range(steps):
  
  with tf.GradientTape() as tape:    
    activations = forward(img)
    loss = calc_loss(activations)
    
  # How cool is this? It's the gradient of the 
  # loss (how excited the layer is) with respect to the
  # pixels of our random image!
  gradients = tape.gradient(loss, img)

  # Normalize the gradients
  gradients /= gradients.numpy().std() + 1e-8 
  
  # Update our image by directly adding the gradients
  # (because they're the same shape!)
  img.assign_sub(gradients)
  
  if step % 10 == 0:
    print ("Step %d, loss %f" % (step, loss))
    show(deprocess(img.numpy()))
    plt.show()

# Let's see the result
# Notice we're calling .numpy() here, which 
# takes us from TensorFlow land -> NumPy land

show(deprocess(img.numpy()))