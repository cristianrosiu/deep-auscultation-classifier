from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
import numpy as np
import pandas as pd

from src.util.plots_helper import plot_confusion_matrix, plot_history, plot_roc


def predict(x_test, best_model_path, history, true=None, survival=False):
    global survival_pred
    if true is None:
        true = [[]]

    # Get best model
    model = load_model(best_model_path)

    # Save predictions
    whistling_pred, rhonchus_pred = model.predict(x_test)
    if survival:
        survival_pred, rhonchus_pred, whistling_pred = model.predict(x_test)
        survival_pred = (survival_pred > 0.5).astype('int32')

    # Process the predictions accordingly.
    whistling_pred = (whistling_pred > 0.5).astype('int32')
    rhonchus_pred = [np.argmax(p) for p in rhonchus_pred]

    # Build classification report
    whistling_report = classification_report(true[0], whistling_pred, output_dict=True)
    rhonchus_report = classification_report(true[1], rhonchus_pred, output_dict=True)

    # Save reports
    whistling_df = pd.DataFrame(whistling_report).transpose()
    rhonchus_df = pd.DataFrame(rhonchus_report).transpose()
    whistling_df.to_csv("../reports/whistling_classification.csv")
    rhonchus_df.to_csv("../reports/rhonchus_classification.csv")

    #if not survival:
    #    plot_history(history, '..plots/learning_curve.png')

    whistling_cm = confusion_matrix(true[0], whistling_pred, normalize='true')
    rhonchus_cm = confusion_matrix(true[1], rhonchus_pred, normalize='true')
    plot_confusion_matrix(whistling_cm, classes=[0, 1], title='Wheezes Confusion Matrix',
                          save_path='../plots/whistling_cm.png')
    plot_confusion_matrix(rhonchus_cm, classes=[0, 1, 2, 3], title='Rhonchus Confusion Matrix',
                          save_path='../plots/rhonchus_cm.png')

    if survival:
        survival_report = classification_report(true[2], survival_pred, output_dict=True)
        survival_df = pd.DataFrame(survival_report).transpose()
        survival_df.to_csv("../reports/survival_classification.csv")

        survival_cm = confusion_matrix(true[2], survival_pred,normalize='true')
        plot_confusion_matrix(survival_cm, classes=[0, 1], title='Confusion Matrix',
                              save_path='../plots/survival_cm.png')



