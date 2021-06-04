# Keras
from keras.layers import Dense, Dropout, Input
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.applications import ResNet50V2
from keras.layers.convolutional import Conv2D, Convolution2D
from keras import Model
from keras.utils.vis_utils import plot_model
from keras import regularizers

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

        x = Conv2D(16, kernel_size=5, padding='same', activation='relu')(visible)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, kernel_size=5, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = GlobalAveragePooling2D()(x)

        output = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(x)
        output1 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(x)

        model = Model(inputs=visible, outputs=[output, output1])
        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'categorical_crossentropy'},
                      loss_weights=self._loss_weights, optimizer=Adam(clipnorm=1., lr=self._lr),
                      metrics=self._test_metrics)

        #plot_model(model, to_file='src/plots/multi_model_plot.png', show_shapes=True, show_layer_names=True)
        model.summary()
        return model

    def build_spectogram_model(self, bands, frames, channels, dropout, fully_connected, filters=24, kernel=3, strides=2, pool=2, kernel_growth=2):
        
        visible = Input(shape=(bands, frames, channels))

        x = Conv2D(filters, kernel, padding='same', strides=strides)(visible)
        x = MaxPooling2D(pool_size=pool)
        x = Activation('relu')
   

        x =Conv2D(filters*kernel_growth, kernel, padding='same', strides=strides)
        x =MaxPooling2D(pool_size=pool)
        x =Activation('relu')
  
       
        x =Conv2D(filters*kernel_growth, kernel, padding='valid', strides=strides)(x)
        x =Activation('relu')(x)
 
    
        x = Flatten()(x)

        x = Dropout(dropout)(x)
        x = Dense(fully_connected, kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')

        x = Dropout(dropout),
        x = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(y)
        x = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(y1)  

        model = Model(inputs=model_input, outputs=[output_whistling, output_rhonchus])

        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'categorical_crossentropy'},
                      loss_weights=self._loss_weights,
                      optimizer=Adam(clipnorm=1., lr=self._lr), metrics=test_metrics)

        return model
