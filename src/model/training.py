from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from multi_models import custom_model, res_model
from features.generate_data import generate_data
from src.model.prediction import predict
from sklearn.model_selection import train_test_split
from src.util.config_helper import read_yaml

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Generate the training configuration
    config = read_yaml('../config.yaml')

    # Set up the weights and number of tasks
    weights = {}
    task_number = 2
    if config['TRAINING']['SURVIVAL']:
        weights = {'whistling': 1e-1, 'rhonchus': 1e-1, 'survival': 1}
        task_number = 3
    else:
        weights = {'whistling': 0.8, 'rhonchus': 0.2}

    # Generate the data
    X, y, y1, y2 = generate_data(pcen=config['TRAINING']['PCEN'])

    # Split data into Training and Test sets
    X_train, X_test, Y_train, Y_test, Y1_train, Y1_test, Y2_train, Y2_test = \
        train_test_split(X, y, y1, y2, test_size=0.1, random_state=42)

    # Get the folds
    kfold = StratifiedKFold(n_splits=config['TRAINING']['SPLITS'], shuffle=True, random_state=42)

    # Save the best model, history based on accuracy
    primary_model = None
    best_history = None
    max_accuracy = -1
    fold = 0
    scores = []

    # KFold Validation
    for train, test in kfold.split(X_train, Y_train, Y1_train):
        # Split the training set into training and validation
        x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test = \
            X_train[train], X_train[test], Y_train[train], Y_train[test], Y1_train[train], Y1_train[test], Y2_train[
                train], Y2_train[test]

        # Reshape train and test data to meet the requirements of a CNN input
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        input_shape = (x_train.shape[1], x_train.shape[2], 1)

        # Build the model
        model = None
        if config['TRAINING']['PCEN']:
            model = res_model(input_shape=input_shape, weights=weights, trainable_layers=0)
        else:
            model = custom_model(input_shape=input_shape, weights=weights, conv_dr=0.2,
                                 survival=config['TRAINING']['SURVIVAL'])

        train_labels, test_labels = {}, {}
        if config['TRAINING']['SURVIVAL']:
            train_labels = {'whistling': y_train, 'rhonchus': y1_train, 'survival': y2_train}
            test_labels = {'whistling': y_test, 'rhonchus': y1_test, 'survival': y2_test}
        else:
            train_labels = {'whistling': y_train, 'rhonchus': y1_train}
            test_labels = {'whistling': y_test, 'rhonchus': y1_test}

        # Fit the model
        file_path = 'saved_models/weights.best.multitask_mfcc{i}.hdf5'.format(i=fold)
        checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)
        history = model.fit(x_train, train_labels, batch_size=config['TRAINING']['BATCH'],
                            epochs=config['TRAINING']['EPOCH'], validation_data=(x_test, test_labels),
                            callbacks=[checkpoint], verbose=1, shuffle=True)

        # Evaluate the model
        score = model.evaluate(x_test, test_labels, verbose=0)
        score = score[-task_number:]

        # Save accuracy of wheezes and rhonchus
        scores.append(score)

        # Find the best model based on max accuracy
        if task_number == 2 and score[0] > max_accuracy:
            max_accuracy = score[0]
            primary_model = file_path
            best_history = history
        elif task_number == 3 and score[2] > max_accuracy:
            max_accuracy = score[2]
            primary_model = file_path
            best_history = history
        fold += 1

    # Save the average accuracies and the standard deviations
    scores = list(zip(*scores))

    results = []
    for score in scores:
        results.append(float("{:.2f}".format(np.mean(list(score)))))
        results.append(float("{:.2f}".format(np.std(list(score)))))

    pd.DataFrame(results).to_csv('../reports/kfolds_report.csv')

    # Build reports and plots
    predict(X_test, best_model_path=primary_model, history=best_history, true_labels=[Y_test, Y1_test, Y2_test],
            survival=config['TRAINING']['SURVIVAL'])
