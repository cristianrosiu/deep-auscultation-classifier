import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from feature_extractor import get_seg_features_df, get_features_df
from sklearn.model_selection import train_test_split
from model import get_model
from plots_helper import plot_kfold, plot_confusion_matrix, plot_roc, plot_history
from util import kfold_split
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
import os, sys
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    def __init__(self, epoch=500, batch=32, n_mfcc=40, sr=None, num_seg=5):
        self.epoch = epoch
        self.batch = batch
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.num_seg = num_seg

def prep_submissions(preds_array, file_name='abc.csv'):
    preds_df = pd.DataFrame(preds_array)
    predicted_labels = preds_df.idxmax(axis=1) #convert back one hot encoding to categorical variabless
    return predicted_labels

def extract_label(label_list, pred_array):
    pred_max = pred_array.argmax()
    return label_list[pred_max]

if __name__ == "__main__":
    data = pd.read_csv('single_task_network/labels_int.csv')
    # Create the configuration of models.
    config = Config(epoch=900, batch=32, n_mfcc=40, num_seg=5,sr=None)

    # Get DataFrame with features
    features_df = get_seg_features_df(data, sampling_rate=None, n_mfcc=config.n_mfcc,label_row='whistling', num_seg=config.num_seg)

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(features_df.feature.tolist())
    y = features_df.class_label.tolist()
    y = np.array([int(lab) for lab in y])

    # Encode the classification labels
   # le = LabelEncoder()
    #yy = to_categorical(le.fit_transform(y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reshape the train and test sets in CNN specific format.
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    
    kfold = kfold_split(X_train, Y_train, n_splits=5)
    #save the model history in a list after fitting so that we can plot later
    model_history = []   
    cvscores = []
    for train, test in kfold:
        x_train, x_test, y_train, y_test = X_train[train], X_train[test], Y_train[train], Y_train[test]

        # Reshape the train and test sets in CNN specific format.
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        model = get_model((X.shape[1], X.shape[2], 1), num_labels=1, binary=True)
        # Training
        checkpointer = ModelCheckpoint(filepath='single_task_network/saved_models/weights.best.basic_single_task.hdf5', verbose=1,
                                       save_best_only=True)

        history = (model.fit(x_train, y_train, batch_size=config.batch, epochs=config.epoch,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer], verbose=1, shuffle=True))
        
        scores = model.evaluate(x_test, y_test, verbose=0)
        cvscores.append(scores[1]*100)
    
    #plot_history(history=history, i=1)

    #scores = model.evaluate(X_test, Y_test, verbose=0)
 
    print(np.mean(cvscores), np.std(cvscores))
    # Plot Learning graph
    plot_kfold(model_history, save_path = 'single_task_network/plots/learning_plot_whistling.png')

    # Plot Confussion Matrix
    #rounded_labels = np.argmax(y_test, axis=1)

    model = load_model('single_task_network/saved_models/weights.best.basic_single_task.hdf5')

    rounded_labels = np.argmax(y_test, axis=1)
    cm = confusion_matrix(rounded_labels, model.predict_classes(x_test))
    plot_confusion_matrix(cm, classes=[0, 1, 3, 4], title='Confusion Matrix', save_path='single_task_network/plots/rhonchus_cm.png')

    # Plot ROC
    y_pred = model.predict(x_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, auc, 'single_task_network/plots/roc_rhonchus.png')

    #preds = (model.predict(x_test) > 0.5).astype("int32")
    preds = np.argmax(model.predict(x_test), axis=-1)

    report = classification_report(y_test, preds, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv('single_task_network/rhonchus_classification_report.csv')
