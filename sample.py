import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from absl import app
from absl import flags
from model import Encoder, Decoder
from data_manager import DataManager

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
FLAGS = flags.FLAGS


def sample(model):
    manager = DataManager(FLAGS.sample_size) # just charge <sample_size> images
    
    Y, X = manager.get_batch(FLAGS.sample_size,np.random.randint(manager.training_set_size // FLAGS.sample_size))
    X_pred = model.predict(X)
    dim = (X_pred[0].shape[0], X_pred[0].shape[1]*FLAGS.sample_size)
    
    _,axes = plt.subplots(3,1)
    axes[0].imshow(np.reshape((np.pad(X,((0,0),(48,48),(48,48),(0,0)),'constant', constant_values=(254))).swapaxes(0,1), dim),cmap='gray')
    axes[0].set_title("X")
    axes[0].axis('off')
    
    axes[1].imshow(np.reshape(Y.swapaxes(0,1), dim),cmap='gray')
    axes[1].set_title("Y")
    axes[1].axis('off')
    
    axes[2].imshow(np.reshape(X_pred.swapaxes(0,1), dim),cmap='gray')
    axes[2].set_title("predict")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(FLAGS.model)
    model = tf.keras.models.load_model(model_path)

    # holds dimensions of latent vector once we find it
    z_dim = None

    # define encoder
    encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # load encoder weights and get the dimensions of the latent vector
    for i, layer in enumerate(model.layers):
        encoder.layers[i] = layer
        if layer.name == "encoder_output":
            z_dim = (layer.get_weights()[0].shape[-1])
            break

    # define encoder
    decoder_in  = tf.keras.Input(shape=(z_dim,))
    decoder_out = Decoder(decoder_in)
    decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # load decoder weights
    found_decoder_weights = False
    decoder_layer_cnt = 0
    for i, layer in enumerate(model.layers):
        print(layer.name)
        weights = layer.get_weights()
        if len(layer.get_weights()) > 0:
            print(weights[0].shape, weights[1].shape)
        if "decoder_input" == layer.name:
            found_decoder_weights = True
        if found_decoder_weights:
            decoder_layer_cnt += 1
            print("dec:" + decoder.layers[decoder_layer_cnt].name)
            decoder.layers[decoder_layer_cnt].set_weights(weights)

    encoder.summary()
    decoder.summary()

    return encoder, decoder, model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    _, _, autoencoder = load_model()
    sample(autoencoder)

if __name__ == '__main__':
    app.run(main)
