"""
Reading tfrecord file and preprocessing it to be ready to feed to mode.fit 
"""
import tensorflow as tf

BATCH_SIZE = 2
WIDTH, HEIGHT = 300, 300
tfrecord_file = "/home/arun/renjith/Renjith/weather_classification/tfrecord/sample.tfrecords"

# train data generator

"""
To create an input pipeline, you must start with a data source. For example, 
to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() 
or tf.data.Dataset.from_tensor_slices(). Alternatively, if your input data is stored 
in a file in the recommended TFRecord format, you can use tf.data.TFRecordDataset().
"""
train_filenames = [tfrecord_file]
val_filenames = [tfrecord_file]

def read_tfrecord(example_):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example_, features)
    image = tf.image.decode_png(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [HEIGHT,WIDTH, 3])
    
    class_label = tf.cast(example['label'], tf.int32)
    
    return image, class_label
 
 
def get_batched_dataset(filenames):
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
    dataset = dataset.prefetch(2) 
    #the maximum number of elements that will be buffered when prefetching
    return dataset

def get_training_dataset():
    return get_batched_dataset(train_filenames)
 
def get_validation_dataset():
    return get_batched_dataset(val_filenames)

get_training_dataset()

