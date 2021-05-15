import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from feature_extractor import get_seg_features_df
from plots_helper import plot_history
from model import get_model
from util import kfold_split
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    def __init__(self, epoch=500, batch=32, n_mfcc=20, sr=None, num_seg=5):
        self.epoch = epoch
        self.batch = batch
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.num_seg = num_seg

if __name__ == "__main__":
    data = pd.read_csv('labels.csv')
    # Create the configuration of models.
    config = Config(epoch=1000, batch=32, n_mfcc=40, num_seg=5,sr=None)

    # Get DataFrame with features
    features_df = get_seg_features_df(data, sampling_rate=None, n_mfcc=config.n_mfcc,label_row='whistling', num_seg=config.num_seg)

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(features_df.feature.tolist())
    y = features_df.class_label.tolist()
    y = np.array([int(lab) for lab in y])

    kfold = kfold_split(X, y, n_splits=2)
    cvscores = []
    for train, test in kfold:

        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # Reshape the train and test sets in CNN specific format.
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        model = get_model((X.shape[1], X.shape[2], 1), num_labels=1, binary=True)

        # Training
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_rhonchus.hdf5', verbose=1,
                                    save_best_only=True)

        history = model.fit(x_train, y_train, batch_size=config.batch, epochs=config.epoch,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpointer], verbose=1, shuffle=True)
        
        scores = model.evaluate(x_test, y_test, verbose=0)
        cvscores.append(scores[1]*100)

        #plot_history(history, 1)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
