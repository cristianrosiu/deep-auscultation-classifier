# Keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation, \
    GlobalMaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras import backend as K
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

WHISTLING_NODES = 1
RHONCHUS_NODES = 4


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


class MultiModel(object):
    def __init__(self, input_shape, test_metrics, loss_weights, lr=0.0001, drop_rate=0.25):
        self._input_shape = input_shape
        self._test_metrics = test_metrics
        self._drop_rate = drop_rate
        self._loss_weights = loss_weights
        self._lr = lr

    def get_custom_model(self):
        visible = Input(shape=self._input_shape)

        x = Conv2D(32, (3, 3), padding="same")(visible)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = GlobalAveragePooling2D()(x)

        output = Dense(64)(x)
        output = Activation("relu")(output)
        output = Dropout(0.5)(output)

        output1 = Dense(64)(x)
        output1 = Activation("relu")(output1)
        output1 = Dropout(0.5)(output1)

        y = Dense(WHISTLING_NODES, activation='sigmoid', name='whistling')(output)
        y1 = Dense(RHONCHUS_NODES, activation='softmax', name='rhonchus')(output1)

        model = Model(inputs=visible, outputs=[y, y1])
        model.compile(loss={'whistling': 'binary_crossentropy', 'rhonchus': 'sparse_categorical_crossentropy'},
                      loss_weights=self._loss_weights, optimizer=Adam(lr=self._lr),
                      metrics=self._test_metrics)

        # plot_model(model, to_file='src/plots/multi_model_plot.png', show_shapes=True, show_layer_names=True)
        model.summary()
        return model

    def build_spectrogram_model(self):
        # load model and specify a new input shape for images
        model_input = Input(shape=self._input_shape)
        base_model = ResNet50(include_top=False, weights=None, input_shape=self._input_shape)

        x = base_model(model_input)
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
                      optimizer=Adam(clipnorm=1.), metrics=self._test_metrics)

        return model
