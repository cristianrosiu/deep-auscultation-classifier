import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from feature_extractor import get_seg_features_df
import os, sys
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    binary = True
    data = pd.read_csv('labels.csv')
    # Create the configuration of models.
    config = Config(epoch=500, batch=32, n_mfcc=40, num_seg=5,sr=None)

    # Get DataFrame with features
    features_df = get_seg_features_df(data, sampling_rate=None, n_mfcc=config.n_mfcc,label_row='whistling', num_seg=config.num_seg)

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
