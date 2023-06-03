import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from datetime import datetime
from tqdm import tqdm
from absl import app
from absl import flags
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from model import AutoEncoder
from data_manager import DataManager

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_integer("epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 10, "batch size")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", True, "continue training same weights")
flags.DEFINE_boolean("keep_best", True, "only save model if it got the best loss")
FLAGS = flags.FLAGS

best_loss = np.inf
model_path = None

def train(model):
    manager = DataManager() 
    loss = [None]*FLAGS.epochs
    batches = manager.training_set_size // FLAGS.batch_size
    
    for epoch in range(FLAGS.epochs):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()
        
        for i in tqdm(range(batches)):
            Y, X = manager.get_batch(FLAGS.batch_size,i)            
            loss[epoch] = model.train_on_batch(X, Y)
               
        print("Epoch {} - loss: {}".format(epoch, loss[epoch]))
        save_model(model, epoch, loss[epoch])
    print("Finished training.")
    
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def save_model(model, epoch, loss):
    """Write logs and save the model"""
    train_summary_writer = tf.summary.create_file_writer(summary_path)
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)

    # save model
    global best_loss
    if not FLAGS.keep_best: 
        model.save(model_path)
    elif loss < best_loss:
        best_loss = loss
        model.save(model_path)

def load_model():
    model = AutoEncoder()

    if os.path.isfile(model_path):
        print("Loading model from", model_path)
        model.load_weights(model_path)

    model.compile(Adam(FLAGS.learning_rate), MSE)
    model.summary()
    return model

def setup_paths():
    """Create log and trained_model dirs. """
    global model_path, summary_path
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs("./trained_model", exist_ok=True)
    timestamp = 'timestamp' # str(datetime.now())

    if FLAGS.keep_training and os.listdir(FLAGS.logdir):
        files = filter(os.path.isdir, glob.glob(FLAGS.logdir + "/*"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        timestamp = os.path.basename(os.path.normpath(list(reversed(files))[0]))

    model_path = os.path.join("./trained_model/DAE-model-" + timestamp + ".h5")
    summary_path = os.path.join(FLAGS.logdir, timestamp)

def main(argv):
    setup_paths()
    model = load_model()
    train(model)

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
