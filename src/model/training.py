from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from multi_models import custom_model, res_model
from features.generate_data import generate_data

import numpy as np
import pandas as pd

SURVIVAL = False
# Models Configuration
weights = {'whistling': 1, 'rhonchus': 1e-1}
if SURVIVAL:
    weights = {'whistling': 1e-1, 'rhonchus': 1e-1, 'survival': 1}

BATCH = 32
EPOCH = 200
N_SPLITS = 5

if __name__ == '__main__':
    path_to_model = 'saved_models/weights.best.multitask_mfcc.hdf5'

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cvscores_whistling = []
    cvscores_rhonchus = []

    primary_model = None
    best_history = None
    max_accuracy = -1
    fold = 0

    X, y, y1, _ = generate_data()

    for train, test in kfold.split(X, y, y1):
        x_train, x_test, y_train, y_test, y1_train, y1_test = X[train], X[test], y[train], y[test], y1[train], y1[test]

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        model = custom_model(input_shape=input_shape, weights=weights, conv_dr=0.2)

        file_path = 'saved_models/weights.best.multitask_mfcc{i}.hdf5'.format(i=fold)
        checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)
        history = model.fit(x_train, {'whistling': y_train, 'rhonchus': y1_train}, batch_size=BATCH,
                            epochs=EPOCH, validation_data=(x_test, {'whistling': y_test, 'rhonchus': y1_test}),
                            callbacks=[checkpoint], verbose=1, shuffle=True)
        # evaluate the model
        scores = model.evaluate(x_test, [y_test, y1_test], verbose=0)
        cvscores_whistling.append(scores[3] * 100)
        cvscores_rhonchus.append(scores[4] * 100)

        if scores[3] > max_accuracy:
            max_accuracy = scores[3]
            primary_model = file_path
            best_history = history
        fold += 1

    w_accuracy = float("{:.2f}".format(np.mean(cvscores_whistling)))
    r_accuracy = float("{:.2f}".format(np.mean(cvscores_rhonchus)))
    w_dev = float("{:.2f}".format(np.std(cvscores_whistling)))
    r_dev = float("{:.2f}".format(np.std(cvscores_rhonchus)))

    scores = [w_accuracy, w_dev, r_accuracy, r_dev]

    pd.DataFrame(scores).to_csv('../reports/kfolds.csv')
    print("whistling: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_whistling), np.std(cvscores_whistling)))
    print("rhonchus: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_rhonchus), np.std(cvscores_rhonchus)))