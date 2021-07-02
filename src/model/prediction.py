from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
import numpy as np
import pandas as pd

from src.util.plots_helper import plot_confusion_matrix, plot_history, plot_roc


def predict(x_test, best_model_path, history, true_labels=None, survival=False):
    """
    This function is used to get the results of the prediction on test data.
    It builds and saves the classification reports and all the validation plots: confusion matrices,
    learning curves and ROC curves

    :param x_test: The test data used for prediction
    :param best_model_path: Path to the wieghts of the model that achieved highest accuracy during training
    :param history: History of the model that achieved highest accuracy during training
    :param true_labels: 2D Array containing the ground truth labels of each task
    :param survival: Checks if in the prediction we should count surival as a task
    """

    global survival_pred
    if true_labels is None:
        true_labels = [[]]

    # Get best model
    model = load_model(best_model_path)

    # Save predictions
    if survival:
        whistling_pred, rhonchus_pred, survival_pred = model.predict(x_test)
        survival_pred = (survival_pred > 0.5).astype('int32')
    else:
        whistling_pred, rhonchus_pred = model.predict(x_test)
    # Process the predictions accordingly.
    whistling_pred = (whistling_pred > 0.5).astype('int32')
    rhonchus_pred = [np.argmax(p) for p in rhonchus_pred]

    # Build classification report
    whistling_report = classification_report(true_labels[0], whistling_pred, output_dict=True)
    rhonchus_report = classification_report(true_labels[1], rhonchus_pred, output_dict=True)

    # Save classification reports reports
    whistling_df = pd.DataFrame(whistling_report).transpose()
    rhonchus_df = pd.DataFrame(rhonchus_report).transpose()
    whistling_df.to_csv("../reports/whistling_classification.csv")
    rhonchus_df.to_csv("../reports/rhonchus_classification.csv")

    # Plot and save the confusion matrices
    whistling_cm = confusion_matrix(true_labels[0], whistling_pred, normalize='true')
    rhonchus_cm = confusion_matrix(true_labels[1], rhonchus_pred, normalize='true')
    plot_confusion_matrix(whistling_cm, classes=[0, 1], title='Wheezes Confusion Matrix',
                          save_path='../plots/confusion/whistling_cm.png')
    plot_confusion_matrix(rhonchus_cm, classes=[0, 1, 2, 3], title='Rhonchus Confusion Matrix',
                          save_path='../plots/confusion/rhonchus_cm.png')

    if survival:
        survival_report = classification_report(true_labels[2], survival_pred, output_dict=True)
        survival_df = pd.DataFrame(survival_report).transpose()
        survival_df.to_csv("../reports/survival_classification.csv")

        survival_cm = confusion_matrix(true_labels[2], survival_pred, normalize='true')
        plot_confusion_matrix(survival_cm, classes=[0, 1], title='Confusion Matrix',
                              save_path='../plots/survival_cm.png')
