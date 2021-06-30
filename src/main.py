# Custom
import numpy as np
from util.plots_helper import plot_confusion_matrix, plot_roc

from src.features.feature_extractor import get_features_df
from util.config import Config
from util.data_split import split_data
from src.model.multi_models import MultiModel

# Keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# sklearn
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve

# Math
import pandas as pd

SPECTROGRAM = False

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/labels_int.csv')
    augmented_data = pd.read_csv('data/augmented.csv')

    config = Config(epoch=1000, batch=32, n_mfcc=40, num_seg=5, sr=None)

    path_to_model = 'src/saved_models/weights.best.multitask.hdf5'

    # Extract features
    features_df = get_features_df(data, sampling_rate=config._sr, n_mfcc=config._n_mfcc,
                                  num_seg=config._num_seg, pcen=False, split=True)
    augmented_df = get_features_df(augmented_data, sampling_rate=config._sr, n_mfcc=config._n_mfcc,
                                   num_seg=config._num_seg, pcen=False, augment=True)
    augmented_df = augmented_df.sample(frac=1)

    X_train, X_test, Y_train, Y_test, Y1_train, Y1_test = split_data(data=features_df, augmented_data=None)

    scores = {'Whistling Error Loss': [],
              'Rhonchus Error Loss': [],
              'Whistling Accuracy': [],
              'Rhonchus Accuracy': []}
    history = []

    '''kf = kfold_split(X_train, Y_train, Y1_train, n_splits=5)
    for train, test in kf:
        # Split the training into fold_training and fold_validation
        x_train, x_test, y_train, y_test, y1_train, y1_test = \
            X_train[train], X_train[test], Y_train[train], Y_train[test], Y1_train[train], Y1_train[test]

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        # Get the model
        test_metrics = {'whistling': 'accuracy', 'rhonchus': 'accuracy'}
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        multi_model = MultiModel(input_shape=input_shape, test_metrics=test_metrics, drop_rate=0.25, lr=0.0001,
                                 loss_weights={'whistling': 0.9, 'rhonchus': 0.2})
        model = multi_model.get_custom_model()
        if SPECTROGRAM:
            model = multi_model.build_spectrogram_model()

        # Training
        checkpoint = ModelCheckpoint(filepath=path_to_model, verbose=1, save_best_only=True)

        history.append(model.fit(x_train, {'whistling': y_train, 'rhonchus': y1_train}, batch_size=config._batch,
                                 epochs=config._epoch, validation_data=(x_test, [y_test, y1_test]),
                                 callbacks=[checkpoint], verbose=1, shuffle=True,use_multiprocessing=True, workers=8))

        loss, main_loss, aux_loss, main_acc, aux_acc = model.evaluate(x=x_test,
                                                                      y={'whistling': y_test, 'rhonchus': y1_test})

        scores['Whistling Error Loss'].append(main_loss * 100)
        scores['Whistling Accuracy'].append(main_acc * 100)
        scores['Rhonchus Error Loss'].append(aux_loss * 100)
        scores['Rhonchus Accuracy'].append(aux_acc * 100)'''

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # Get the model
    test_metrics = {'whistling': 'accuracy', 'rhonchus': 'accuracy'}
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    multi_model = MultiModel(input_shape=input_shape, test_metrics=test_metrics, drop_rate=0.25, lr=0.0001,
                             loss_weights={'whistling': 1, 'rhonchus': 1e-1})

    model = multi_model.get_custom_model()
    if SPECTROGRAM:
        model = multi_model.build_spectrogram_model()

    # Training
    checkpoint = ModelCheckpoint(filepath=path_to_model, verbose=1, save_best_only=True)

    history.append(model.fit(X_train, {'whistling': Y_train, 'rhonchus': Y1_train}, batch_size=config._batch,
                             epochs=config._epoch, validation_data=(X_test, [Y_test, Y1_test]),
                             callbacks=[checkpoint], verbose=1, shuffle=True))

    loss, main_loss, aux_loss, main_acc, aux_acc = model.evaluate(x=X_test,
                                                                  y={'whistling': Y_test, 'rhonchus': Y1_test})

    scores['Whistling Error Loss'].append(main_loss * 100)
    scores['Whistling Accuracy'].append(main_acc * 100)
    scores['Rhonchus Error Loss'].append(aux_loss * 100)
    scores['Rhonchus Accuracy'].append(aux_acc * 100)
    # Load model
    model = load_model(path_to_model)

    # Generate prediction
    pred = model.predict(X_test)
    whistling_pred = pred[0]
    rhonchus_pred = pred[1]

    whistling_pred = np.argmax(whistling_pred, axis=-1)
    rhonchus_pred = [np.argmax(p) for p in rhonchus_pred]
    Y1_test = [np.argmax(p) for p in Y1_test]

    # Generate reports
    whistling_report = classification_report(Y_test, whistling_pred, output_dict=True)
    rhonchus_report = classification_report(Y1_test, rhonchus_pred, output_dict=True)
    whistling_df = pd.DataFrame(whistling_report).transpose()
    rhonchus_df = pd.DataFrame(rhonchus_report).transpose()
    df_scores = pd.DataFrame(scores)
    df_scores.loc['mean'] = df_scores.mean()

    # Save reports
    df_scores.to_csv("reports/training_scores.csv")
    whistling_df.to_csv("reports/whistling_classification.csv")
    rhonchus_df.to_csv("reports/rhonchus_classification.csv")

    # Plot Learning graph
    #plot_history(history, i=0, save_path='plots/learning_plot.png')
    #plot_kfold(history, save_path='plots/rhonchus_learning_plot.png', label='Rhonchus')
    #plot_kfold(history, save_path='plots/whistling_learning_plot.png', label='Whistling')

    # Plot Confussion Matrix
    whistling_cm = confusion_matrix(Y_test, whistling_pred)
    rhonchus_cm = confusion_matrix(Y1_test, rhonchus_pred)
    plot_confusion_matrix(whistling_cm, classes=[0, 1], title='Confusion Matrix',
                          save_path='plots/whistling_cm.png')
    plot_confusion_matrix(rhonchus_cm, classes=[0, 1, 2, 3], title='Confusion Matrix',
                          save_path='plots/rhonchus_cm.png')

    # Plot ROC
    y_pred = model.predict(X_test)[0].ravel()
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
    auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, auc, 'plots/whistling_roc.png')
