# Keras
from keras.layers import Dense, Dropout, Input
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.applications import ResNet50V2
from keras.layers.convolutional import Conv2D
from keras import Model

WHISTLING_NODES = 1
RHONCHUS_NODES = 4


class MultiModel(object):
    def __init__(self, input_shape, test_metrics, loss_weights, lr=0.01, drop_rate=0.25):
        self._input_shape = input_shape
        self._test_metrics = test_metrics
        self._drop_rate = drop_rate
        self._loss_weights = loss_weights
        self._lr = lr

    def get_custom_model(self, batch_normalization=False):
        visible = Input(shape=self._input_shape)
        
        x = Conv2D(32, kernel_size=3, padding='same')(visible)
        x = Activation('relu')(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(self._drop_rate)(x)

        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Activation('relu')(x) 
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(self._drop_rate)(x)

        x = GlobalAveragePooling2D()(x)

        output = Dense(64, activation='relu')(x)
        output = Dropout(0.5)(output)

        output1 = Dense(64, activation='relu')(x)
        output1 = Dropout(0.5)(output1)

        output = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(output)
        output1 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(output1)

        model = Model(inputs=visible, outputs=[output, output1])
        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'categorical_crossentropy'},
                      loss_weights=self._loss_weights, optimizer=Adam(clipnorm=1., lr=self._lr),
                      metrics=self._test_metrics)
        model.summary()
        return model

    def get_pretrained_model(self):
        model_input = Input(shape=self._input_shape)
        base_model = ResNet50V2(weights=None, input_tensor=model_input, include_top=False)

        for layer in base_model.layers[:]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        y = Dense(64, activation='relu')(x)
        y = Dropout(0.5)(y)

        y1 = Dense(64, activation='relu')(x)
        y1 = Dropout(0.5)(y1)

        output_whistling = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(y)
        output_rhonchus = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(y1)

        model = Model(inputs=model_input, outputs=[output_whistling, output_rhonchus])

        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'categorical_crossentropy'},
                      loss_weights=self._loss_weights,
                      optimizer=Adam(clipnorm=1., lr=self._lr), metrics=test_metrics)

        return model
