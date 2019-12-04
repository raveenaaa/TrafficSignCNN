import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil
from timeit import default_timer as timer

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

import os
import tensorflow as tf

# Change these
choice = 0
activation = "sigmoid"
dropout = 0.5
optimizer = "adam"
neuron = 128

epochs = 15


# Change these: ends
target_accuracies = [0.947426762,0.905463183,0.90783848,0.914014252,0.940617577,0.946634996,0.9338,0.93111639,0.938796516]
target_accuracy = target_accuracies[choice]
data_filename = "./data/data{}.pickle".format(choice)

os.makedirs("./results_step3/test", exist_ok=True)
os.makedirs("./results_step3/train_validate", exist_ok=True)
os.makedirs("./saved_models_step3", exist_ok=True)
os.makedirs("././training_plots_step3", exist_ok=True)


pkl_filename = "./saved_models_step3/data{}_model.pkl".format(choice)

plot_acc = "./training_plots_step3/data{}_acc.png".format(choice)
plot_loss = "./training_plots_step3/data{}_loss.png".format(choice)

file_training_validation_results = "./results_step3/train_validate/data{}_train_validate_results.txt".format(choice)
file_testing_results = "./results_step3/test/data{}_test_results.txt".format(choice)



epoch_step_size = 3



annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
mc = ModelCheckpoint('classifier.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

models = {}
history = {}


def plot_model(history, epoch):
  plt.figure()
  plt.plot(history.history['acc'], 'b', label='Training Accuracy')
  plt.plot(history.history['val_acc'], 'r', label='Validation Accuracy')
  plt.legend(loc='upper right')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.xticks(np.arange(0,epochs,step=epoch_step_size))
  plt.title('Accuracy Curves for {0} epochs'.format(epoch))
  plt.savefig(plot_acc, bbox_inches='tight')

  plt.figure()
  plt.plot(history.history['loss'], 'b', label='Training Loss')
  plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.xticks(np.arange(0,epochs,step=epoch_step_size))
  plt.title('Loss Curves for {0} epochs'.format(epoch))
  plt.savefig(plot_loss, bbox_inches='tight')

def create_model(activation='tanh', dropout=0.0,optimizer='adam',neurons=64, channel=1): 
  model = Sequential()
  model.add(Conv2D(neurons, kernel_size=3, padding='same', activation=activation, input_shape=(32, 32, channel)))
  model.add(MaxPool2D(pool_size=2))
  model.add(Dropout(dropout))

  model.add(Flatten())
  model.add(Dense(256, activation=activation))
  model.add(Dense(256, activation=activation))
  # Add more layers start here
  model.add(Dense(2048, activation=activation))
  model.add(Dense(1024, activation=activation))
  model.add(Dropout(dropout))
  model.add(Dense(512, activation=activation))
  model.add(Dropout(dropout))
  model.add(Dense(512, activation=activation))
  # Add more layers ends here
  model.add(Dense(43, activation='softmax'))
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# Execution starts here

with tf.device('/device:GPU:0'):
  with open(data_filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')  # dictionary type

  # Preparing y_train and y_validation for using in Keras
  data['y_train'] = to_categorical(data['y_train'], num_classes=43)
  data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
  data['y_test'] = to_categorical(data['y_test'],num_classes = 43)

  # Making channels come at the end
  data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
  data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
  data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)


  CHANNEL = data['x_train'].shape[-1]       # 1 for grayscale, 3 for RGB


  data["x_train"] = np.concatenate((data["x_train"],data["x_validation"]),axis=0)
  data["y_train"] = np.concatenate((data["y_train"],data["y_validation"]),axis=0)

  training_file = open(file_training_validation_results, "w+")

  model = create_model(activation=activation,dropout=dropout,optimizer=optimizer,neurons=neuron, channel=CHANNEL)
  model.fit(data['x_train'], data['y_train'],batch_size=512, epochs = epochs,validation_split=0.3,callbacks=[annealer, mc])

  training_accuracies = model.history.history["acc"]
  validation_accuracies = model.history.history["val_acc"]
  optimal_epochs = len(training_accuracies)
  
  train_result = "Training Accuracies = "+str(training_accuracies)+"\nValidation Accuracies = "+str(validation_accuracies)+"\n"
  training_file.write(train_result + "\n========\n")

  training_file.close()

  plot_model(model.history,epochs)
  
  """# Calculating accuracy with testing dataset"""

  predictions = model.predict(data['x_test'])
  y_test = [ np.argmax(t) for t in data['y_test'] ]
  y_predict = [ np.argmax(t) for t in predictions ]
  test_accuracy = accuracy_score(y_test,y_predict)

  test_result_file_handle = open(file_testing_results, "w+")
  test_result = "BEST MODEL\nTest Accuracy = {0} for {1} epochs".format(test_accuracy,optimal_epochs)
  if test_accuracy > target_accuracy:
    test_result+="\nAccuracy improved from step 2"
  else:
    test_result+="\nAccuracy worsened from step 2"
  test_result_file_handle.write(test_result)
  test_result_file_handle.close()

  with open(pkl_filename, 'wb') as file:
      pickle.dump(model, file)
