from keras.layers import Dense, Dropout, Input
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from keras.applications import VGG16
from keras import Model
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.layers.convolutional import Conv2D
from config import Config
from feature_extractor import get_features_df
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from util import kfold_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

WHISTLING_NODES = 1
RHONCHUS_NODES = 4


def multi_model_custom(input_shape, drop_rate, test_metrics):
    visible = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(visible)
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drop_rate)(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drop_rate)(x)

    x = GlobalAveragePooling2D()(x)

    output = Dense(64, activation='relu')(x)
    output = Dropout(0.5)(output)

    output1 = Dense(64, activation='relu')(x)
    output1 = Dropout(0.5)(output1)

    output = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(x)
    output1 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(x)

    model = Model(inputs=visible, outputs=[output, output1])
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), loss_weights=[0.8, 1], metrics=test_metrics)

    return model


def multi_model(input_shape, drop_rate, test_metrics):
    base_model = VGG16(weights=None, input_shape=input_shape, include_top=False)

    for layer in base_model.layers[:]:
        layer.trainable = False

    model_input = Input(shape=input_shape)
    x = base_model(model_input)
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(drop_rate)(x)

    y1 = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(x)
    y2 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(x)

    model = Model(inputs=model_input, outputs=[y1, y2])

    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=0.0001),
                  metrics=test_metrics)

    return model


if __name__ == '__main__':
    data = pd.read_csv('labels_int.csv')

    config = Config(epoch=100, batch=32, n_mfcc=40, num_seg=5, sr=None)

    features_df = get_features_df(data, sampling_rate=config._sr, n_mfcc=config._n_mfcc, num_seg=config._num_seg)

    X = np.array(features_df.feature.tolist())
    Y = features_df.class_labels.tolist()

    y = np.array([labels[0] for labels in Y])
    y1 = [labels[1] for labels in Y]

    # Encode the classification labels
    le = LabelEncoder()
    y1 = to_categorical(le.fit_transform(y1))

    #X_train, X_test, y_train, y_test, y1_train, y1_test = train_test_split(X, y, y1, test_size=0.2, random_state=42)

    kfold = kfold_split(X, y, y1, n_splits=2)
    cvscores = []

    for train, test in kfold:
        X_train, X_test, y_train, y_test, y1_train, y1_test = X[train], X[test], y[train], y[test], y1[train], y1[test]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        test_metrics = {'whistling': 'accuracy', 'rhonchus': 'accuracy'}
        model = multi_model_custom((X.shape[1], X.shape[2], 1), 0.25, test_metrics)

        # Training
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.multitask.hdf5', verbose=1,
                                       save_best_only=True)

        model.fit(X_train, [y_train, y1_train], batch_size=config._batch, epochs=config._epoch,
                  validation_data=(X_test, [y_test, y1_test]),
                  callbacks=[checkpointer], verbose=1, shuffle=True)

        loss, main_loss, aux_loss, main_acc, aux_acc = model.evaluate(x=X_test,
                                                                      y={'whistling': y_test, 'rhonchus': y1_test})
        cvscores.append([main_acc * 100, aux_acc*100])

        pred = model.predict(X_test)[0]
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        print(classification_report(y_test, pred))

    print(cvscores)

