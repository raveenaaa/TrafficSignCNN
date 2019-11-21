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

import os
import tensorflow as tf

data_filename = "data8.pickle"
pkl_filename = "data8_model.pkl"

with tf.device('/device:GPU:0'):


  with open(data_filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')  # dictionary type

  # Preparing y_train and y_validation for using in Keras
  data['y_train'] = to_categorical(data['y_train'], num_classes=43)
  data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)

  # Making channels come at the end
  data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
  data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
  data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)


  data["x_train"] = np.concatenate((data["x_train"],data["x_validation"]),axis=0)
  data["y_train"] = np.concatenate((data["y_train"],data["y_validation"]),axis=0)
  # Showing loaded data from file
  for i, j in data.items():
      if i == 'labels':
          print(i + ':', len(j))
      else: 
          print(i + ':', j.shape)


  def create_model(activation='tanh', dropout=0.0,optimizer='adam',neurons=64): 
    model = Sequential()
    model.add(Conv2D(neurons, kernel_size=3, padding='same', activation=activation, input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(250, activation=activation))
    model.add(Dense(250, activation=activation))
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



  annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
  epochs = 10

  activations = ["sigmoid","tanh","relu"]
  dropouts = [0.1,0.3,0.5]
  optimizers = ['adam','sgd']
  neurons = [32, 64, 128]
  models = {}
  history = {}

  es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=2, verbose=1, restore_best_weights=True)
  mc = ModelCheckpoint('classifier.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

  for activation in activations:
    models[activation] = {}
    for dropout in dropouts:
      models[activation][dropout] = {}
      for optimizer in optimizers:
        models[activation][dropout][optimizer] = {}
        for neuron in neurons:
          model = create_model(activation=activation,dropout=dropout,optimizer=optimizer,neurons=neuron)
          model.fit(data['x_train'], data['y_train'],batch_size=512, epochs = epochs,validation_split=0.3,callbacks=[annealer, es, mc])
          models[activation][dropout][optimizer][neuron] = model

  """# Calculating accuracy with testing dataset"""

  best_activation = -1
  best_dropout = -1
  best_optimizer = ''
  best_neuron = -1
  best_acc = 0
  for activation in activations:
    for dropout in dropouts:
      for optimizer in optimizers:
        for neuron in neurons:
          temp = models[activation][dropout][optimizer][neuron].predict(data['x_test'])
          temp = np.argmax(temp, axis=1)

          temp = np.mean(temp == data['y_test'])
          if temp > best_acc:
            best_acc = temp
            best_activation = activation
            best_dropout = dropout
            best_optimizer = optimizer
            best_neuron = neuron
          print("Test Accuracy = {0} for the model: Activation={1}, Dropout={2}, Optimizer={3}, Neurons={4}".format(temp,activation,dropout,optimizer,neuron))

  print("BEST MODEL\nTest Accuracy = {0} for the model: Activation={1}, Dropout={2}, Optimizer={3}, Neurons={4}".format(best_acc,best_activation,best_dropout,best_optimizer,best_neuron))

  with open(pkl_filename, 'wb') as file:
      pickle.dump(models[best_activation][best_dropout][best_optimizer][best_neuron], file)

