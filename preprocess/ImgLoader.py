import Constants
import time
import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import tqdm

class ImgLoader(object):

    def _bytes_feature(self, value):
        """Convert to tfrecord bytes features"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_image_records(self):
        """Loop through the image files and write to tfrecords"""
        writer = tf.python_io.TFRecordWriter(Constants.record_file)
        for folder in tqdm.tqdm(Constants.file_list):
            for character in glob.glob(folder + "/*"):
                for file_name in glob.glob(character + "/*.png"):
                    im = Image.open(file_name)
                    np_img = np.array(im)

                    key = file_name.split("images_background/", 1)[1]
                    key = key[0:key.rfind('/')]

                    key_raw = str.encode(key)
                    img_raw = np_img.tostring()

                    feature={
                        'image_raw': self._bytes_feature(img_raw),
                        'key_raw': self._bytes_feature(key_raw)}

                    example = tf.train.Example(features=tf.train.Features(feature = feature))
                    writer.write(example.SerializeToString())

        writer.close()

    def rec_img(self):
        """Reconstruct the image and show the image, total images: 19280"""

        record_iterator = tf.python_io.tf_record_iterator(path=Constants.record_file)
        sum = 0

        for string_record in record_iterator:

            sum +=1

            example = tf.train.Example()
            example.ParseFromString(string_record)

            img_string = (example.features.feature[Constants.id_record_image]
                .bytes_list
                .value[0])

            label = (example.features.feature[Constants.id_record_key]
                .bytes_list
                .value[0])

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((Constants.image_dimension, Constants.image_dimension))
            img = Image.fromarray(reconstructed_img, 'L')
            # img.show()

        print(sum)




