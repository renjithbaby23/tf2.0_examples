"""
reference: https://androidkt.com/feed-tfrecord-to-keras/
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import glob

DATA_DIR = "data/"
TFREC_PREFIX = "tfrecords/sample"
CLASSES = ['1', '2']
BATCH_SIZE = 4
SHARD_SIZE = 100
DESIRED_WIDTH = 300
DESIRED_LENGTH = 300

# Converting the values into features

# _int64 is used for numeric values liek labels
def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytestring_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# for floats
def _float_feature(value): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# function to read image and label
def read_image_and_label(img_path):
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)
    image = tf.image.resize(image, [DESIRED_LENGTH, DESIRED_WIDTH])
    label = tf.strings.split(img_path, sep='/')
    label = tf.strings.split(label[-1], sep='_')

    return image,label[0]


def to_tfrecord(img_bytes, label):  
    """
    Function to create protocol buffers.
    Arguments:
        img_bytes: input image as bytestring
        label: input label
    """
    class_num = np.argmax(np.array(CLASSES)==label) 
    feature = {
      "image": _bytestring_feature(img_bytes), # one image in the list
      "class": _int_feature(class_num),        # one class in the list      
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def recompress_image(image, label):
    """
    Recompressing the read image into jpeg format
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=False, chroma_downsampling=False)
    return image, label

dataset = tf.data.Dataset.list_files('data/*/*.jpeg', seed=10000) # This also shuffles the data
dataset = dataset.map(read_image_and_label)
dataset = dataset.map(recompress_image, num_parallel_calls=None)
dataset = dataset.batch(SHARD_SIZE) 

# iterating through the dataset and writing to tfrecord file
for shard, (image, label) in enumerate(dataset):
    shard_size = image.numpy().shape[0]
    filename = TFREC_PREFIX + "{:02d}-{}.tfrec".format(shard, shard_size)

    with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
            example = to_tfrecord(image.numpy()[i],label.numpy()[i])
            out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))



def read_tfrecord(example):
    """
    This is the read operation that will be performed on the input tfrecord datasets.
    The features will be parsed as per the details below. 
    Feel free to modify the features dictionary as per the encoding.
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [DESIRED_LENGTH, DESIRED_WIDTH, 3])
    
    class_label = tf.cast(example['class'], tf.int32)
    
    return image, class_label
 

def get_batched_dataset(filenames):
    """
    get batches of dataset, arguments:
    filenames : list of tfrecord files
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=None)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=None)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
    dataset = dataset.prefetch(2) #

    return dataset

# use glob(TFREC_PREFIX + "*.tfrec") to get list of all filenames
training_filenames = ["tfrecords/sample00-10.tfrec"]
def get_training_dataset():
    return get_batched_dataset(training_filenames)
 
validation_filenames = ["tfrecords/sample00-10.tfrec"]
def get_validation_dataset():
    return get_batched_dataset(validation_filenames)

# Just checking if the files generated are correct
x = get_training_dataset()
count = 0
for i in x:
    print("batch shape: ", i[0].shape)
    count += 1
    if count > 0:
        break
