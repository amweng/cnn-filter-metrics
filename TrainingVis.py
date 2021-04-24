import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import keras


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
        f1score = 2*(precision*recall)/(precision+recall)


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
            plt.title("Distribution of prediction probabilities at epoch no. {}".format(epoch), 
                      fontsize=16)
            plt.hist(preds, bins=50,edgecolor='k')
            
            plt.figure(figsize=(10,3))
            plt.title("Loss over epoch")
            plt.plot(N, self.losses)
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax = ax.ravel()
            ax[0].plot(N, self.precision, label = "Precision", c='red')
            ax[1].plot(N, self.recall, label = "Recall", c='red')
            ax[2].plot(N, self.f1score, label = "F1 score", c='red')
            ax[0].set_title("Precision at Epoch No. {}".format(epoch))
            ax[1].set_title("Recall at Epoch No. {}".format(epoch))
            ax[2].set_title("F1-score at Epoch No. {}".format(epoch))
            ax[0].set_xlabel("Epoch #")
            ax[1].set_xlabel("Epoch #")
            ax[2].set_xlabel("Epoch #")
            ax[0].set_ylabel("Precision")
            ax[1].set_ylabel("Recall")
            ax[2].set_ylabel("F1 score")
            ax[0].set_ylim(0,1)
            ax[1].set_ylim(0,1)
            ax[2].set_ylim(0,1)
            
            plt.show()