'''
Training Script for VDCNN Text
'''
import keras
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import losses
import keras.backend as K
import numpy as np
from absl import flags
import h5py
import math
import sys
import datetime
from sklearn.model_selection import train_test_split
from vdcnn import *
from data_helper import *
import custom_callbacks
from keras.optimizers import Adam
from keras.optimizers import Nadam


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
# Parameters settings
# Data loading params

tf.flags.DEFINE_string("database_path", "mytrain/data/", "Path for the dataset to be used.")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 1024, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("pool_type", "max", "Types of downsampling methods, use either three of max (maxpool), k_max (k-maxpool) or conv (linear) (default: 'max')")
tf.flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("shortcut", True, "Use optional shortcut (default: False)")
tf.flags.DEFINE_boolean("sorted", False, "Sort during k-max pooling (default: False)")
tf.flags.DEFINE_boolean("use_bias", False, "Use bias for all conv1d layers (default: False)")

# Training parameters
flags.DEFINE_integer("batch_size", 70, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 100)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on test set after this many steps (default: 100)")



FLAGS = flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
print("-"*20)
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

data_helper = data_helper(sequence_max_length=FLAGS.sequence_length)

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    train_data, train_label, test_data, test_label = data_helper.load_dataset(FLAGS.database_path)
    print("Loading data succees...")

    return train_data, train_label, test_data, test_label

def train(x_train, y_train, x_test, y_test):
    # Init Keras Model here
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    model = VDCNN(num_classes=1,
                  depth=FLAGS.depth, 
                  sequence_length=FLAGS.sequence_length, 
                  shortcut=FLAGS.shortcut,
                  pool_type=FLAGS.pool_type, 
                  sorted=FLAGS.sorted, 
                  use_bias=FLAGS.use_bias)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam= Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model_json = model.to_json()
    with open("vdcnn_model.json","w") as json_file:
        json_file.write(model_json)                    # Save model architecture
    time_str = datetime.datetime.now().isoformat()
    print("{}: Model saved as json.".format(time_str))
    print("")

    # Trainer
    # Tensorboard and extra callback to support steps history
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=50, batch_size=FLAGS.batch_size, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath="logs/checkpoints/vdcnn_weights_val_acc_{val_acc:.4f}.h5", period=1,
                                   verbose=1, save_best_only=True, mode='max', monitor='val_acc')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

    loss_history = custom_callbacks.loss_history(model, tensorboard)
    evaluate_step = custom_callbacks.evaluate_step(model, checkpointer, tensorboard, FLAGS.evaluate_every, FLAGS.batch_size, x_val, y_val)
    testcallback= custom_callbacks.TestCallback(model,(x_test, y_test))

    # Fit model
    model.fit(x_train, y_train, batch_size=FLAGS.batch_size, epochs=FLAGS.num_epochs, validation_data=(x_val, y_val), shuffle= True,
              verbose=1, callbacks=[checkpointer, tensorboard, loss_history, evaluate_step, testcallback])
    print('-'*30)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Done training.".format(time_str))
    K.clear_session()
    print('-'*30)
    print()

if __name__=='__main__':
    x_train, y_train, x_test, y_test = preprocess()
    train(x_train, y_train, x_test, y_test)
