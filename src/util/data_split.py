# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

# Keras
from keras.utils import to_categorical

# Math
import numpy as np


def kfold_split(X, y, y1, n_splits=5, seed=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)
    return kfold.split(X, y, y1)


def split_data(features_df):
    x = np.array(features_df.feature.tolist())
    Y = features_df.class_labels.tolist()

    y = np.array([labels[0] for labels in Y])
    y1 = [labels[1] for labels in Y]

    # Encode the classification labels
    le = LabelEncoder()
    y1 = to_categorical(le.fit_transform(y1))

    x_train, x_test, y_train, y_test, y1_train, y1_test = train_test_split(x, y, y1, test_size=0.33, random_state=42)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, x_test, y_train, y_test, y1_train, y1_test
