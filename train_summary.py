# Load various imports 
from datetime import datetime
from os import listdir
from os.path import isfile, join

import librosa

import numpy as np
import pandas as pd

from tensorflow.keras.losses import CategoricalCrossentropy 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Input, Activation
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

global features_final
global labels_final

def prepareDataset():
    ncol1, ncol2, ncol3 = 1, 20,862

    features_final = np.memmap('memmapped.dat', dtype=np.float32,
                mode='w+', shape=(ncol1, ncol2, ncol3))
    i1=1
    while i1 < 13:
        with open('../train_dataset1/eval_features_p'+str(i1)+'.npz', 'rb') as f1:
            aux = np.load(f1, mmap_mode='r+')
            features_final= np.concatenate((features_final,aux['a']))
            i1 +=1


    ncol1, ncol2 = 1, 2

    labels_final = np.memmap('memmapped.dat', dtype=np.float32,
                mode='w+', shape=(ncol1, ncol2))

    i1=1
    while i1 < 13:
        with open('../train_dataset1/eval_labels_p'+str(i1)+'.npz', 'rb') as f1:
            aux = np.load(f1, mmap_mode='r+')
            labels_final = np.concatenate((labels_final,aux['a']))
            i1 +=1

    features_final=features_final[1:]
    labels_final=labels_final[1:]


    # add channel dimension for CNN
    features_final = np.reshape(features_final, (*features_final.shape,1))

    print(len(features_final))
    print(features_final.shape)
    print(len(labels_final))
    print(labels_final.shape)

    return features_final,labels_final



# # Test
def train_test_split(features_final, labels_final):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(features_final, labels_final, stratify=labels_final, test_size=0.2, random_state = 42)
    return x_train, x_test, y_train, y_test

# ### Instantiate Model vars

class My_Custom_Generator(Sequence) :
  
    def __init__(self, features, labels, batch_size) :
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(features_final.shape[0] / float(self.batch_size))).astype(np.int)
  

    def __getitem__(self, idx) :
        batch_x = self.features[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

def train(x_train, x_test, y_train, y_test,labels_final):
    batch_size = 32

    my_training_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

    # ORIGINAL
    num_rows = 20
    num_columns = 862
    num_channels = 1

    num_labels = labels_final.shape[1]
    print(num_labels)
    filter_size = 2

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(num_rows, num_columns, num_channels)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_labels, activation='softmax'))

    # ### Fit the model

    print(len(x_train))
    print(len(x_test))

    # Compile the model
    # Train standard
    epochs = 70
    batch_size = 32
    verbose = 1
    # optimizer = optimizers.SGD(lr=0.002, momentum=0.9, nesterov=True)
    optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)


    model.compile(loss=CategoricalCrossentropy(),
                optimizer=optimizer,
                metrics=['accuracy'])


    callbacks = [
        ModelCheckpoint(
            filepath='mymodel2_{epoch:02d}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1)
    ]
    model.fit_generator(generator=my_training_batch_generator,steps_per_epoch = int( len(x_train) / batch_size),epochs = epochs,verbose = verbose,validation_data = my_validation_batch_generator,validation_steps = int(len(x_test)/ batch_size),callbacks=callbacks)


# ### Test model with the Dataset used for the training phase

    model.summary()
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    accuracy = 100*score[1]
    print("Pre-training accuracy: %.4f%%" % accuracy)


if __name__ == "__main__":
    features_final,labels_final = prepareDataset()
    x_train, x_test, y_train, y_test = train_test_split(features_final,labels_final)
    train(x_train, x_test, y_train, y_test, labels_final)





