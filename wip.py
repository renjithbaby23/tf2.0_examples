"""
reference: https://androidkt.com/feed-tfrecord-to-keras/
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import glob

class1_dir = "data/class1"
class2_dir = "data/class2"
tfrecord_filename = 'sample.tfrecords'

class1_dir = "/home/arun/renjith/FCA/data/night/raw_images/no_ramp/"
class2_dir = "/home/arun/renjith/FCA/data/night/raw_images/ramp/"
tfrecord_filename = '/home/arun/renjith/Renjith/weather_classification/tfrecord/sample.tfrecords'

DESIRED_WIDTH = 300
DESIRED_LENGTH = 300

# Converting the values into features
# _int64 is used for numeric values

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Initiating the writer and creating the tfrecords file.
writer = tf.io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

class1_imgs = glob.glob(class1_dir + '*.png')
class2_imgs = glob.glob(class2_dir + '*.png')

print("len(class1_imgs): ", len(class1_imgs))
print("len(class2_imgs): ", len(class2_imgs))

images = class1_imgs + class2_imgs

# Generating tfrecord file for 10 files
for img_path in images[:10]:

    bits = tf.io.read_file(img_path)
    img = tf.image.decode_png(bits)
    img = tf.image.resize(img, [DESIRED_LENGTH, DESIRED_WIDTH])

    # label=0 for class1 image and label=1 for class2 image
    label = 0 if 'class1' in img_path else 1
    feature = { 'label': _int64_feature(label),
              'image': _bytes_feature(img.numpy().tostring()) }

    # print(img.shape, label)
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example.

    writer.write(example.SerializeToString())
writer.close()

"""
If you closely see the process involved, it’s very simple.

Data -> FeatureSet -> Example -> Serialized Example -> tfRecord.
"""

# Reading a tfrecord file

"""
So to read it back, the process is reversed.

tfRecord -> SerializedExample -> Example -> FeatureSet -> Data

"""

filenames = "/home/arun/renjith/Renjith/weather_classification/tfrecord/sample.tfrecords"

readed_record = tf.data.TFRecordDataset(filenames)


feature_description = { 'image': tf.io.FixedLenFeature([], tf.string),
                        'label': tf.io.FixedLenFeature([], tf.int64)}
  
def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = readed_record.map(_parse_function)
print("parsed_dataset: ", parsed_dataset)

# Reading and displaying one image from the tfrecord file
for parsed_record in parsed_dataset.take(1):
    label = parsed_record['label'].numpy()
    image = parsed_record['image'].numpy()
    image = tf.io.decode_raw(image, tf.int32)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [DESIRED_LENGTH, DESIRED_WIDTH, 3])
    print("*"*20)
    print("label = ",label)
    print(image.numpy().shape)
    print("*"*20)
    image_ = image.numpy().astype(np.uint8)
    image_ = Image.fromarray(image_)
    image_.show()

