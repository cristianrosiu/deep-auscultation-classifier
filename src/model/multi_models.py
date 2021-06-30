from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Input, Dense, \
    BatchNormalization, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

WHISTLING_NODES = 1
RHONCHUS_NODES = 4
SURVIVAL_NODES = 1


def custom_model(input_shape, weights=None, conv_dr=0.2, survival=False):
    if weights is None:
        weights = {'whistling': 1, 'rhonchus': 1}

    model_input = Input(shape=input_shape)

    x = Conv2D(16, kernel_size=2)(model_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(conv_dr)(x)

    x = Conv2D(32, kernel_size=2)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(conv_dr)(x)

    x = Conv2D(64, kernel_size=2)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(conv_dr)(x)

    x = Conv2D(128, kernel_size=2)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(conv_dr)(x)

    output_whistling = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(GlobalAveragePooling2D()(x))
    output_rhonchus = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(GlobalAveragePooling2D()(x))
    if survival:
        output_survival = Dense(SURVIVAL_NODES, activation='sigmoid', name='survival')(GlobalAveragePooling2D()(x))

    model = Model(inputs=model_input, outputs=[output_whistling, output_rhonchus])
    if survival:
        model = Model(inputs=model_input, outputs=[output_whistling, output_rhonchus, output_survival])

    model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'sparse_categorical_crossentropy'},
                  loss_weights=weights,
                  optimizer=Adam(learning_rate=0.0005), metrics='accuracy')
    if survival:
        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'sparse_categorical_crossentropy', 'survival':'binary_crossentropy'},
                      loss_weights=weights,
                      optimizer=Adam(learning_rate=0.0005), metrics='accuracy')
    model.summary()

    return model


def res_model(input_shape, trainable_layers, weights=None, add_dense=False):
    if weights is None:
        weights = {'whistling': 1, 'rhonchus': 1}

    # load model and specify a new input shape for images
    model_input = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    x = base_model(model_input)
    x = GlobalAveragePooling2D()(x)

    if add_dense:
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        y = Dense(128, activation='relu')(x)
        y = Dropout(0.5)(y)
        y = Dense(64, activation='relu')(y)
        y = Dropout(0.5)(y)

        y1 = Dense(128, activation='relu')(x)
        y1 = Dropout(0.5)(y1)
        y1 = Dense(64, activation='relu')(y1)
        y1 = Dropout(0.5)(y1)

    output_whistling = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(x)
    output_rhonchus = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(x)
    if add_dense:
        output_whistling = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(y)
        output_rhonchus = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(y1)

    model = Model(inputs=model_input, outputs=[output_whistling, output_rhonchus])

    model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'sparse_categorical_crossentropy'},
                  loss_weights=weights,
                  optimizer='adam', metrics='accuracy')
    model.summary()

    return model
