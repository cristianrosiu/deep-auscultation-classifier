from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from multi_models import custom_model, res_model
from features.generate_data import generate_data
from src.model.prediction import predict
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Model's Configuration
SURVIVAL = False
PCEN = False
weights = {'whistling': 1, 'rhonchus': 1}
if SURVIVAL:
    weights = {'whistling': 1e-1, 'rhonchus': 1e-1, 'survival': 1}
BATCH = 32
EPOCH = 20
N_SPLITS = 2

if __name__ == '__main__':
    # Get data
    X, y, y1, y2 = generate_data(pcen=True)

    X_train, X_test, Y_train, Y_test, Y1_train, Y1_test, Y2_train, Y2_test=\
        train_test_split(X, y, y1, y2, test_size=0.1, random_state=42)

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cvscores_whistling = []
    cvscores_rhonchus = []

    # Save the best model, history based on accuracy
    primary_model = None
    best_history = None
    max_accuracy = -1
    fold = 0

    # KFold Validation
    for train, test in kfold.split(X_train, Y_train, Y1_train):
        x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test = \
            X_train[train], X_train[test], Y_train[train], Y_train[test], Y1_train[train], Y1_train[test], Y2_train[train], Y2_train[test]

        # Reshape train and test data to meet the requirments of a CNN input
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        input_shape = (x_train.shape[1], x_train.shape[2], 1)

        # Build the model
        model = custom_model(input_shape=input_shape, weights=weights, conv_dr=0.2)
        if PCEN:
            model = res_model(input_shape=input_shape, weights=weights)

        file_path = 'saved_models/weights.best.multitask_mfcc{i}.hdf5'.format(i=fold)
        checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)

        history = model.fit(x_train, {'whistling': y_train, 'rhonchus': y1_train}, batch_size=BATCH,
                            epochs=EPOCH, validation_data=(x_test, {'whistling': y_test, 'rhonchus': y1_test}),
                            callbacks=[checkpoint], verbose=1, shuffle=True)
        if SURVIVAL:
            history = model.fit(x_train,
                                {'whistling': y_train, 'rhonchus': y1_train, 'survival': y2_train},
                                batch_size=BATCH, epochs=EPOCH, validation_data=
                                (x_test, {'whistling': y_test, 'rhonchus': y1_test, 'survival': y2_test}),
                                callbacks=[checkpoint], verbose=1, shuffle=True)
        # Evaluate the model
        scores = model.evaluate(x_test, [y_test, y1_test], verbose=0)
        if SURVIVAL:
            scores = model.evaluate(x_test, [y_test, y1_test, y2_test], verbose=0)
        # Save accuracy of wheezes and rhonchus
        cvscores_whistling.append(scores[3] * 100)
        cvscores_rhonchus.append(scores[4] * 100)

        # Find the best model based on max accuracy
        if scores[3] > max_accuracy:
            max_accuracy = scores[3]
            primary_model = file_path
            best_history = history
        fold += 1

    # Save the average accuracies and the standard deviations
    w_accuracy = float("{:.2f}".format(np.mean(cvscores_whistling)))
    r_accuracy = float("{:.2f}".format(np.mean(cvscores_rhonchus)))
    w_dev = float("{:.2f}".format(np.std(cvscores_whistling)))
    r_dev = float("{:.2f}".format(np.std(cvscores_rhonchus)))

    scores = [w_accuracy, w_dev, r_accuracy, r_dev]
    pd.DataFrame(scores).to_csv('../reports/kfolds.csv')

    predict(X_test, best_model_path=primary_model, history=best_history, true=[Y_test, Y1_test, Y2_test], survival=SURVIVAL)