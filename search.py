import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from absl import app
from absl import flags
from model import Encoder
from data_manager import DataManager

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_integer("sample_size", 5, "images to found")
flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
FLAGS = flags.FLAGS


def search(encoder):
    manager = DataManager()
    
    #             select an image 32x32
    # encoder need a numpy array with shape (None,32,32,1)
    img_path = input("Enter the name and the extension of the image : ")
    if os.path.exists("./celeba/"+img_path):
        img = np.array([cv2.resize(cv2.cvtColor(cv2.imread("./celeba/"+img_path),cv2.COLOR_BGR2GRAY),(32,32))])
    else :
        print("Image <"+img_path+"> not found !")
        print("A random image from DB will be use instead.")
        r = np.random.randint(manager.training_set_size)
        img = manager.X[r:r+1]
    
    # compute the loss of the result of the encoded image selected and all the encoded images from dataset
    loss = np.array(tf.keras.metrics.mean_squared_error(encoder(img), encoder(manager.X)))
    indices = (loss.argsort())[0:FLAGS.sample_size] # permutation array
    
    loss = loss[indices]
    imgs = manager.X[indices]
    
    # plot each image of the result with its loss
    _,axes = plt.subplots(2,FLAGS.sample_size)
    for i in range(FLAGS.sample_size):
        if i == 0:
            axes[0][i].imshow(np.reshape(img,(32,32)),cmap='gray')
            axes[0][i].set_title("original")
            axes[0][i].axis('off')
        else :
            axes[0][i].set_visible(False)
        axes[1][i].imshow(np.reshape(imgs[i],(32,32)),cmap='gray')
        axes[1][i].set_title(loss[i])
        axes[1][i].axis('off')
    plt.tight_layout()
    plt.show()

def load_encoder():
    model = tf.keras.models.load_model(os.path.abspath(FLAGS.model))
    
    encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    for i, layer in enumerate(model.layers):
        if layer.name == "decoder_input":
            break
        encoder.layers[i] = layer
        weights = layer.get_weights()
        encoder.layers[i].set_weights(weights)
    
    return encoder
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    search(load_encoder())

if __name__ == '__main__':
    app.run(main)