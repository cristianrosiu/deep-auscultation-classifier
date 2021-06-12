# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

# Keras
from tensorflow.keras.utils import to_categorical

# Math
import numpy as np


def kfold_split(X, y, y1, n_splits=5, seed=42):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold.split(X, y, y1)


def split_data(data, augmented_data=None):
    global augmented_x, augmented_y, augmented_y1
    le = LabelEncoder()

    if augmented_data is not None:
        augmented_x = np.array(augmented_data.feature.tolist())
        augmented_Y = augmented_data.class_labels.tolist()
        augmented_y = np.array([labels[0] for labels in augmented_Y])
        augmented_y1 = [labels[1] for labels in augmented_Y]

    x = np.array(data.feature.tolist())
    Y = data.class_labels.tolist()

    y = np.array([labels[0] for labels in Y])
    y1 = np.array([labels[1] for labels in Y])
    #y1 = to_categorical(le.fit_transform(y1))

    x_train, x_test, y_train, y_test, y1_train, y1_test = train_test_split(x, y, y1, test_size=0.2, random_state=52)

    if augmented_data is not None:
        x_train = np.append(x_train, augmented_x, axis=0)
        y_train = np.append(y_train, augmented_y)
        y1_train = np.append(y1_train, augmented_y1)

    # Encode the classification labels for rhonchus
    #y1_train, y1_test = to_categorical(le.fit_transform(y1_train)), to_categorical(le.fit_transform(y1_test))
    # Reshape test data
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, x_test, y_train, y_test, y1_train, y1_test
