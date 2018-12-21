from preprocess import ImgLoader
import Constants
import DataContainer
import tensorflow as tf

ldr = ImgLoader.ImgLoader()
# ldr.rec_img()

print(tf.__version__)
Dc = DataContainer.DataContainer()
Dc.visualise_data()