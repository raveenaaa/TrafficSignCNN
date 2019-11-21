import tensorflow as tf
import pickle
import numpy as np
import math as math
import matplotlib.pyplot as plt

# Opening file for reading in binary mode
with open('./data/data7.pickle', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # dictionary type


# Preparing y_train and y_validation for using in Keras
data['y_train'] = tf.keras.utils.to_categorical(data['y_train'], num_classes=43)
data['y_validation'] = tf.keras.utils.to_categorical(data['y_validation'], num_classes=43)

# Making channels come at the end
data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

# Showing loaded data from file
for i, j in data.items():
    if i == 'labels':
        print(i + ':', len(j))
    else:
        print(i + ':', j.shape)

# Preparing function for ploting set of examples
# As input it will take 4D tensor and convert it to the grid
# Values will be scaled to the range [0, 255]
# def convert_to_grid(x_input):
#     N, H, W, C = x_input.shape
#     grid_size = int(math.ceil(math.sqrt(N)))
#     grid_height = H * grid_size + 1 * (grid_size - 1)
#     grid_width = W * grid_size + 1 * (grid_size - 1)
#     grid = np.zeros((grid_height, grid_width, C)) + 255
#     next_idx = 0
#     y0, y1 = 0, H
#     for y in range(grid_size):
#         x0, x1 = 0, W
#         for x in range(grid_size):
#             if next_idx < N:
#                 img = x_input[next_idx]
#                 low, high = np.min(img), np.max(img)
#                 grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
#                 next_idx += 1
#             x0 += W + 1
#             x1 += W + 1
#         y0 += H + 1
#         y1 += H + 1
#
#     return grid
#
#
# # Visualizing some examples of training data
# examples = data['x_train'][:81, :, :, :]
# print(examples.shape)  # (81, 32, 32, 3)
#
# # Plotting some examples
# fig = plt.figure()
# grid = convert_to_grid(examples)
# plt.imshow(grid.astype('uint8').squeeze(), cmap='gray')
# plt.axis('off')
# plt.gcf().set_size_inches(15, 15)
# plt.title('Some examples of training data', fontsize=18)
#
# # Showing the plot
# plt.show()
#
# # Saving the plot
# fig.savefig('training_examples.png')








def create_model(activation='tanh', dropout=0.3,optimizer='adam',neurons=128):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(neurons, kernel_size=3, padding='same', activation=activation, input_shape=(32, 32, 1)))
  model.add(tf.keras.layers.MaxPool2D(pool_size=2))
  model.add(tf.keras.layers.Dropout(dropout))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation=activation))
  model.add(tf.keras.layers.Dense(128, activation=activation))
  model.add(tf.keras.layers.Dense(43, activation='softmax'))
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model




model = create_model()
EPOCHS = 10
BATCH_SIZE = 512

with tf.device('/device:GPU:0'):

    # Random Rotations
    # define data preparation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45, brightness_range=[0.2,1.0])
    # fit parameters from data
    datagen.fit(data['x_train'])

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(data['x_train'], data['y_train'], batch_size=BATCH_SIZE),
                        steps_per_epoch=len(data['x_train']) / BATCH_SIZE, epochs=EPOCHS)


    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(data['x_train'], data['y_train'], batch_size=16):
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].squeeze(), cmap=plt.get_cmap('gray'))
        # show the plot
        plt.show()
        break

    # here's a more "manual" example
    for e in range(EPOCHS):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(data['x_train'], data['y_train'], batch_size=BATCH_SIZE):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(data['x_train']) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break









