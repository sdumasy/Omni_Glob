import Constants
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class DataContainer(object):

    def __init__(self):
        """splits the data"""
        self.train_files = [Constants.record_file]
        dataset = self.get_dataset()
        self.train_data = dataset.take(Constants.train_size)
        self.dev_data = dataset.take(Constants.dev_size)
        self.test_data = dataset.take(Constants.test_size)

        self.train_iterator = self.train_data.make_initializable_iterator(shared_name=None)
        self.dev_iterator = self.train_data.make_initializable_iterator(shared_name=None)
        self.test_iterator = self.train_data.make_initializable_iterator(shared_name=None)

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.train_files)
        dataset = dataset.repeat()  # repeat indefinitely
        dataset = dataset.shuffle(Constants.train_size)
        dataset = dataset.map(self.parse_function)
        dataset = dataset.batch(Constants.batch_size)
        """Initiliase the data container with the feature names and amount of samples"""
        return dataset

    def parse_function(self, example):
        """Parse the tfreocrds and store them into a queue with batches"""

        features = tf.parse_single_example(
            example, features={
                Constants.id_record_key : tf.FixedLenFeature([], tf.string),
                Constants.id_record_image : tf.FixedLenFeature([], tf.string),
                Constants.id_record_label : tf.FixedLenFeature([], tf.int64),
            })

        img = tf.decode_raw(features[Constants.id_record_image], tf.uint8)
        img = tf.reshape(img, [Constants.image_dimension, Constants.image_dimension, 1])
        # img = tf.to_float(img)
        img = tf.image.per_image_standardization(img)
        # img = tf.nn.l2_normalize(img)
        # img = tf.contrib.layers.instance_norm(img, epsilon=1e-06)
        # img = tf.image.per_image_standardization(img)

        key = tf.cast(features[Constants.id_record_key], tf.string)
        label = tf.cast(features[Constants.id_record_label], tf.int64)

        return key, img, label

    def visualise_data(self):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            sess.run(self.train_iterator.initializer)

            for i in range(10):

                key, img, label = sess.run(self.train_iterator.get_next())
                print("KEY: ", key[0], "LABEL: ", label[0])
                plt.imshow(img[0].reshape(Constants.image_dimension, Constants.image_dimension), interpolation='nearest')
                plt.show()

            sess.close()