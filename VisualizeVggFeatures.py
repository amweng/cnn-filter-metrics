import sys, os
from matplotlib import pyplot
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import math



### Code References: https://towardsdatascience.com/how-to-create-custom-real-time-plots-in-deep-learning-ecbdb3e7922f
### Code References: https://www.analyticsvidhya.com/blog/2020/11/tutorial-how-to-visualize-feature-maps-directly-from-cnn-layers/
### Code References: Visualizing intermediate activation in Convolutional Neural Networks with Keras

class VisualizeVggFeatures(keras.callbacks.Callback):

    def on_test_end(self,logs={}):

        model = self.model

        num_layers = 6
            
        randomNoise = np.random.rand(1,32,32,3)
        randomNoise = randomNoise * 255
        print(randomNoise.shape)


        layer_outputs = [layer.output for layer in model.layers[:num_layers]] # Gathers the outputs of the layers we want
        activation_model = Model(inputs=model.input, outputs=layer_outputs) # Isolates the model layers from our model
        activations = activation_model.predict(randomNoise) # Returns a list of five Numpy arrays: one array per layer activation

        images_per_row = 16

        layer_names = []
        for layer in model.layers[:num_layers]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            

        for layer_name, layer_activation in zip(layer_names, activations): # Iterates over every layer
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            output_size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            layer_vis = np.zeros((output_size * n_cols, images_per_row * output_size))


            for col in range(n_cols):
                for row in range(images_per_row):
                    feature = layer_activation[0, :, :, col * images_per_row + row]
                    # Scale and transform the activation for display
                    feature -= feature.mean() # Subtract the mean
                    feature /= feature.std() # Normalize

                    # Don't allow the intensity values to be too large (max 200... over 200 is harsh to look at)
                    feature *= 50
                    feature += 150
                    feature = np.clip(feature, 0, 255).astype('uint8')

                    # displays a panel of 
                    layer_vis[col * output_size : (col + 1) * output_size, 
                                row * output_size : (row + 1) * output_size] = feature
            scale = 1. / output_size
            plt.figure(figsize=(scale * layer_vis.shape[1],
                                scale * layer_vis.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(layer_vis, aspect='auto', cmap='plasma')