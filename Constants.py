import glob

raw_data_directory = '/home/stephan/Documents/Data_01/OmniGlob/images_background/*'
record_output_dir = '/home/stephan/Documents/Data_01/OmniGlob/tfrecords'
train_record_output_dir = record_output_dir + '/record.tfrecords'

file_list = glob.glob(raw_data_directory)
class_list = []

image_dimension = 105