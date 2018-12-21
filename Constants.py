import glob

raw_data_directory = '/home/stephan/Documents/Data_01/OmniGlob/images_background/*'
record_output_dir = '/home/stephan/Documents/Data_01/OmniGlob/tfrecords'
saved_res_model_file = '/home/stephan/Documents/projects/Omni_Glob/saved_models/resmodel/model_ep_'
record_file = record_output_dir + '/record.tfrecords'

file_list = glob.glob(raw_data_directory)
class_list = []

id_record_image = 'image_raw'
id_record_key = 'key_raw'
id_record_label = 'label_raw'

#total images: 19280, 964 differnet characters
output_classes = 964
train_size = 17280
dev_size = 1000
test_size = 1000

image_dimension = 105

batch_size = 32
epochs = 5
n_step_epoch = int(train_size / batch_size)

