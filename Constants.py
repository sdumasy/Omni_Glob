import glob

raw_data_directory = '/home/stephan/Documents/Data_01/OmniGlob/images_background/*'
record_output_dir = '/home/stephan/Documents/Data_01/OmniGlob/tfrecords'
record_file = record_output_dir + '/record.tfrecords'

file_list = glob.glob(raw_data_directory)
class_list = []

id_record_image = 'image_raw'
id_record_key = 'key_raw'

#total images: 19280
train_size = 17280
dev_size = 1000
test_size = 1000

batch_size = 32

image_dimension = 105