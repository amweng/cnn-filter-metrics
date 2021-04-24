# keras imports for the dataset and building our neural network
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils


### CODE FROM: https://towardsdatascience.com/how-to-create-custom-real-time-plots-in-deep-learning-ecbdb3e7922f
### 

class TrainingVis(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.f1score = []
        self.precision = []
        self.recall = []
        self.logs = []

    def on_batch_end(self,batch,logs={}):
        tp = logs.get('tp')
        fp = logs.get('fp')
        fn = logs.get('fn')
        loss = logs.get('loss')

        m = self.model
        preds = m.predict(X_train)
        

  
        precision = tp/(tp+fp+1)
        recall = tp/(tp+fn+1)
        f1score = 2*(precision*recall)/(precision+recall+1)


        self.logs.append(logs)
        self.losses.append(loss)
        self.f1score.append(f1score)
        self.precision.append(precision)
        self.recall.append(recall)

                # Plots every 5th epoch
        if batch > 0 and batch%5==0:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(10,3))
            plt.title("Distribution of prediction probabilities at batch no. {}".format(batch), 
                      fontsize=16)
            plt.hist(preds, bins=50,edgecolor='k')
            
            plt.figure(figsize=(10,3))
            plt.title("Loss over batch")
            plt.plot(N, self.losses)
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax = ax.ravel()
            ax[0].plot(N, self.precision, label = "Precision", c='red')
            ax[1].plot(N, self.recall, label = "Recall", c='red')
            ax[2].plot(N, self.f1score, label = "F1 score", c='red')
            ax[0].set_title("Precision at Batch No. {}".format(batch))
            ax[1].set_title("Recall at Batch No. {}".format(batch))
            ax[2].set_title("F1-score at Batch No. {}".format(batch))
            ax[0].set_xlabel("Batch #")
            ax[1].set_xlabel("Batch #")
            ax[2].set_xlabel("Batch #")
            ax[0].set_ylabel("Precision")
            ax[1].set_ylabel("Recall")
            ax[2].set_ylabel("F1 score")
            ax[0].set_ylim(0,1)
            ax[1].set_ylim(0,1)
            ax[2].set_ylim(0,1)
            
            plt.show()

def compile_train_model(
    model,
    x_train,
    y_train,
    callbacks=None,
    learning_rate=0.001,
    metrics = None,
    class_weight = None,
    batch_size=1,
    epochs=10,
    verbose=0,
):
    """
  Compiles and trains a given Keras model with the given data. 
  Assumes Adam optimizer for this implementation.
  Assumes categorical cross-entropy loss.
  
  Arguments
          learning_rate: Learning rate for the optimizer Adam
          batch_size: Batch size for the mini-batch optimization
          epochs: Number of epochs to train
          verbose: Verbosity of the training process
  
  Returns
  A copy of the model
  """

    model_copy = model
    model_copy.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    if callbacks != None:
        model_copy.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose,
        )
    else:
        model_copy.fit(
            x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=verbose
        )
    return model_copy

# to calculate accuracy
from sklearn.metrics import accuracy_score

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

metrics = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.FalsePositives(name="fp"),
]
plot_metrics = TrainingVis()

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics = metrics,optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=1, callbacks =[plot_metrics], validation_data=(X_test, Y_test))










