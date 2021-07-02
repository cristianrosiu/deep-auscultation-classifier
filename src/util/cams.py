from scipy.ndimage.measurements import label
import keras
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import librosa
from src.features.feature_extractor import extract_pcen, scale_minmax


def get_output_layer(model, layer_name):

    """
    Extracts a layer from a given model.

    :param model: model used
    :param layer_name: layer to be extracted
    :return: the extracted layer
    """
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def get_heatmap(model, layer_index, img, original_img, conv_layer):
    """
    Get the output of a given conv layer and creates a heatmap overlay for
    the original image.
    :param model:
    :param layer_index: Index of output layer to be extracted
    :param img: image of features
    :param original_img: original spectrogram image
    :param conv_layer: Name of the convolutional layer to be used
    :return: Heatmap image overlay
    """

    width, height = original_img.shape[0], original_img.shape[1]

    class_weights = model.layers[layer_index].get_weights()[0]
    final_conv_layer = get_output_layer(model, conv_layer)
    get_output = keras.backend.function([model.layers[0].input],
                                        [final_conv_layer.output, model.layers[layer_index].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])

    for i, w in enumerate(class_weights[np.argmax(predictions), :]):
        cam += w * conv_outputs[i, :, :]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.3)] = 0

    return heatmap


def generate_cam(model, audio_path, rec_name, rhonchus_label, whistling_label):
    """
    Generates a plot which shows the original image and the same image but
    with a class activation map overlay.

    :param model: model to be used
    :param audio_path: path to audio file
    :param rec_name: name of the audio
    :param rhonchus_label: rhonchus label of given audio
    :param whistling_label: whistling label of given audio
    """
    y, sr = librosa.load(audio_path, sr=None)
    mels = extract_pcen(y=y, sr=sr)

    original_img = scale_minmax(mels, 0, 255).astype(np.uint8)
    original_img = np.flip(original_img, axis=0)  # put low frequencies at the bottom in image

    plt.show()
    # Get the color map by name:
    cm = plt.get_cmap('inferno')
    colored_img = cm(original_img)
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

    features = mels.reshape((1, mels.shape[0], mels.shape[1], 1))

    whistling_heatmap = get_heatmap(model=model, layer_index=-2, img=features, original_img=colored_img)
    rhonchus_heatmap = get_heatmap(model=model, layer_index=-1, img=features, original_img=colored_img)

    whistling_cam = (whistling_heatmap * 0.8 + colored_img).astype(np.uint8)
    rhonchus_cam = (rhonchus_heatmap * 0.8 + colored_img).astype(np.uint8)

    fig, ax = plt.subplots(nrows=2, figsize=(8, 10))
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', ax=ax[0])
    ax[0].set(title='Original Image', xlabel=None)
    ax[0].label_outer()
    ax[1].set(title='Class Activation Map')
    ax[1].imshow(whistling_cam)

    plt.savefig('../plots/cams/whistling_{name}_{r}{w}.png'.format(name=rec_name, r=rhonchus_label, w=whistling_label))
    plt.show()

    fig, ax = plt.subplots(nrows=2, figsize=(8, 10))
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', ax=ax[0])
    ax[0].set(title='Original Image', xlabel=None)
    ax[0].label_outer()
    ax[1].set(title='Class Activation Map')
    ax[1].imshow(rhonchus_cam)

    plt.savefig('../plots/cams/rhonchus_{name}_{r}{w}.png'.format(name=rec_name, r=rhonchus_label, w=whistling_label))
    plt.show()