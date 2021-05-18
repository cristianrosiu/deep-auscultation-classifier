from keras.layers import Dense, Dropout, Input
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras import Model
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.layers.convolutional import Conv2D
from config import Config
from feature_extractor import get_features_df
from sklearn.preprocessing import LabelEncoder
from util import kfold_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from keras.optimizers import SGD

WHISTLING_NODES = 1
RHONCHUS_NODES = 4
SPECTOGRAM = True

def multi_model_custom(input_shape, drop_rate, test_metrics):
    visible = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, padding='same')(visible)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drop_rate)(x)

    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drop_rate)(x)

    x = GlobalAveragePooling2D()(x)

    output = Dense(128, activation='relu')(x)
    output = Dropout(0.5)(output)

    output1 = Dense(128, activation='relu')(x)
    output1 = Dropout(0.5)(output1)

    output = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(x)
    output1 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(x)

    model = Model(inputs=visible, outputs=[output, output1])
    model.compile(loss={'whistling':'binary_crossentropy', 'rhonchus':'categorical_crossentropy'}, optimizer=Adam(clipnorm=1., lr=0.00005),
                  metrics=test_metrics)
    return model


def multi_model(input_shape, drop_rate, test_metrics):
    base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in base_model.layers[:]:
        layer.trainable = False

    model_input = Input(shape=input_shape)
    x = base_model(model_input)
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop_rate)(x)

    y1 = Dense(128, activation='relu')(x)
    y1 = Dropout(drop_rate)(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(drop_rate)(y1) 

    y2 = Dense(128, activation='relu')(x)
    y2 = Dropout(drop_rate)(y2)
    y2 = Dense(64, activation='relu')(y2)
    y2 = Dropout(drop_rate)(y2) 

    y1 = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(y1)
    y2 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(y2)

    model = Model(inputs=model_input, outputs=[y1, y2])

    model.compile(loss={'whistling':'binary_crossentropy', 'rhonchus':'categorical_crossentropy'}, optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=test_metrics, loss_weights={'whistling':0.7, 'rhonchus': 0.3})

    return model


if __name__ == '__main__':
    data = pd.read_csv('labels_int.csv')

    config = Config(epoch=200, batch=32, n_mfcc=20, num_seg=5, sr=None)
    
    features_df = get_features_df(data, sampling_rate=config._sr, n_mfcc=config._n_mfcc, num_seg=config._num_seg)
    if SPECTOGRAM:
        features_df = get_features_df(data, sampling_rate=config._sr, n_mfcc=config._n_mfcc, num_seg=config._num_seg, spectogram=True)

    X = np.array(features_df.feature.tolist())
    Y = features_df.class_labels.tolist()

    y = np.array([labels[0] for labels in Y])
    y1 = [labels[1] for labels in Y]

    # Encode the classification labels
    le = LabelEncoder()
    y1 = to_categorical(le.fit_transform(y1))
    
    scores = {'Whistling Error Loss': [],
              'Rhonchus Error Loss': [],
              'Whistling Accuracy': [],
              'Rhonchus Accuracy': []}
    
    kfold = kfold_split(X, y, y1, n_splits=3)

    for train, test in kfold:
        X_train, X_test, y_train, y_test, y1_train, y1_test = X[train], X[test], y[train], y[test], y1[train], y1[test]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)

        test_metrics = {'whistling': 'accuracy', 'rhonchus': 'accuracy'}
    
        model = multi_model_custom((X.shape[1], X.shape[2], 3), 0.25, test_metrics)
        if SPECTOGRAM:
            model = multi_model((X.shape[1], X.shape[2], 3), 0.0, test_metrics)

        # Training
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.multitask.hdf5', verbose=1,
                                       save_best_only=True)

        model.fit(X_train, {'whistling':y_train, 'rhonchus':y1_train}, batch_size=config._batch, epochs=config._epoch,
                  validation_data=(X_test, [y_test, y1_test]),
                  callbacks=[checkpointer], verbose=1, shuffle=True)

        loss, main_loss, aux_loss, main_acc, aux_acc = model.evaluate(x=X_test,
                                                                      y={'whistling': y_test, 'rhonchus': y1_test})

        scores['Whistling Error Loss'].append(main_loss*100)
        scores['Whistling Accuracy'].append(main_acc*100)
        scores['Rhonchus Error Loss'].append(aux_loss*100)
        scores['Rhonchus Accuracy'].append(aux_acc*100)

        pred = model.predict(X_test)[0]
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        report = classification_report(y_test, pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        print(df)

    df_scores = pd.DataFrame(scores)
    df_scores.loc['mean'] = df_scores.mean()
    print(df_scores)