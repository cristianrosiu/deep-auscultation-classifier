import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, \
    BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from feature_extractor import get_features_df, get_seg_features_df
from plots_helper import plot_confusion_matrix, plot_roc, plot_history
import os, sys
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectogram_generator import get_spec_df

class Config:
    def __init__(self, epoch=500, batch=32, n_mfcc=20, sr=None, num_seg=5):
        self.epoch = epoch
        self.batch = batch
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.num_seg = num_seg


def get_model(input_shape, num_labels, binary=False):
    #Construct model
    model = Sequential()

    # model.add(Conv2D(32, kernel_size=2, padding='same', input_shape=input_shape, activation='relu'))
    # model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=2, strides=2))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
    # model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=2, strides=2))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
    # model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=2, strides=2))
    # model.add(Dropout(0.25))

    # model.add(GlobalAveragePooling2D())

    # model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Conv2D(16, kernel_size=3, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    optimizer = Adam(learning_rate=0.0001)
    # Compile the model
    if binary:
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    else:
        model.add(Dense(units=num_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


    model.summary()

    return model


if __name__ == "__main__":
    binary = False
    data = pd.read_csv('labels.csv')
    # Create the configuration of models.
    config = Config(epoch=1000, batch=32, n_mfcc=40, num_seg=4,sr=None)

    # Get DataFrame with features
    #features_df = get_seg_features_df(data, sampling_rate=None, n_mfcc=config.n_mfcc,label_row='rhonchus', num_seg=config.num_seg)
    features_df = get_features_df(data, sampling_rate=None, n_mfcc=config.n_mfcc, label_row='whistling') 
    #features_df = get_spec_df(data, 'rhonchus')
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(features_df.feature.tolist())
    y = features_df.class_label.tolist()
    y = np.array([int(lab) for lab in y])

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
    if binary:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the train and test sets in CNN specific format.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model = get_model((X.shape[1], X.shape[2], 1), num_labels=yy.shape[1], binary=False)
    if binary:
       model = get_model((X.shape[1], X.shape[2], 1), num_labels=1, binary=True)
    
    score = model.evaluate(x_test, y_test, verbose=1)

    # Training
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_rhonchus.hdf5', verbose=1,
                                   save_best_only=True)
    start = datetime.now()

    history = model.fit(x_train, y_train, batch_size=config.batch, epochs=config.epoch,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer], verbose=1, shuffle=True)

    plot_history(history, 1)

    duration = datetime.now() - start

    print("Training completed in time: ", duration)

    training_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)

    print("Training: {tscore}\nTest: {ttscore}".format(tscore=training_score[1], ttscore=test_score[1]))

    # EVALUATION METRICS
    #rounded_labels = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, model.predict_classes(x_test))
    plot_confusion_matrix(cm, classes=[0, 1], title='Confustion Matrix')
    plt.show()

    y_pred = model.predict(x_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, auc)
