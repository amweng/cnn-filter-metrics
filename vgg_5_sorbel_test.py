# baseline model with dropout on the cifar10 dataset
import sys
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD

import tensorflow as tf

import numpy as np
import math

import skimage
from skimage import io, data
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
 

def sobel_filter_dataset(dataset):
    # convert to tensor and [0,1]
    dataset = tf.convert_to_tensor(np.array(dataset[:,:,:,:])/255)
    dataset = tf.image.sobel_edges(dataset)

    sobel_y = np.asarray(dataset[:, :, :, :, 0]) # sobel in y-direction
    sobel_y = np.clip(sobel_y / 4 + 0.5, 0, 1) # remap to [0,1]

    sobel_x = np.asarray(dataset[:, :, :, :, 1]) # sobel in x-direction
    sobel_x = np.clip(sobel_x / 4 + 0.5, 0, 1) # remap to [0,1]

    dataset = np.clip(0.5*sobel_x + 0.5*sobel_y, 0, 1)
    return dataset


(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = sobel_filter_dataset(trainX)
testX = sobel_filter_dataset(testX)
trainY = to_categorical(trainY)
testY = to_categorical(testY)



# define cnn model
def define_model_VGG5():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + 'vgg3_vanilla_plot.png')

	pyplot.close()
 
# CODE MODIFIED FROM tensorflow: Save and load models
def run_test_harness(trainX,testX,trainY,testY):

	from keras.callbacks import ModelCheckpoint, EarlyStopping
	checkpoint = ModelCheckpoint("/Users/andrewweng/developer/cnn-filter-metrics/vgg_5_sobel_demo_checkpoints/vgg5_sobel_demo.chkpt", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
	early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
	# load dataset
	# trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	# trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model_VGG5()
	# fit model
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1, callbacks=[checkpoint,early])
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	return history
 
# entry point, run the test harness
history = run_test_harness(trainX,testX,trainY,testY)
summarize_diagnostics(history,'vgg5_sobel_demo')
