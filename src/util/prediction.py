from scipy.ndimage.measurements import label
from feature_extractor import get_spectogram, get_mfcc, extract_features
import librosa
import numpy as np
from keras.models import load_model
import keras
import cv2


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map(model_path, input, output_path,original_path, input_shape, mfcc=False, sound='whistling'):
    model = load_model(model_path)

    original_img = input

    conv_layer = 'conv5_block3_out'
    if mfcc:
        original_img = get_mfcc(input)
        conv_layer = 'conv2d_3'

    output_idx = -2
    if sound == 'rhonchus':
        output_idx = -1

    img = input.reshape(input_shape)

    width, height = original_img.shape[0], original_img.shape[1]

    class_weights = model.layers[output_idx].get_weights()[0]
    final_conv_layer = get_output_layer(model, conv_layer)
    get_output = keras.backend.function([model.layers[0].input],
                                        [final_conv_layer.output, model.layers[output_idx].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])

    for i, w in enumerate(class_weights[0, :]):
        cam += w * conv_outputs[i, :, :]

    print('predictions', predictions)

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0

    img = heatmap * 0.5 + cv2.applyColorMap(original_img, cv2.COLORMAP_SUMMER)
    if mfcc:
        img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)
    cv2.imwrite(original_path, original_img)


def extract_label(label_list, pred_array):
    pred_max = pred_array.argmax()
    return label_list[pred_max]


def get_prediction(file_path, model_path, sampling_rate=None, mfcc=False):
    """
    :param file_path: Path to the audio
    :param model: The model to be used for prediction
    :param n_mfcc: The number of coefficients to be extracted
    :param sampling_rate: Sampling rate
    :param model_path: Path to the model's weights
    :return: Predicted label of the audio (i.e: 0 (OK) 1 (MILD) 2(MODERATE) 3(SEVERE)
    """
    # settings
    hop_length = 512  # number of samples per time-step in spectrogram
    n_mels = 128  # number of bins in spectrogram. Height of image

    audio, sr = librosa.load(file_path, sr=sampling_rate, res_type='kaiser_best')

   
    # Extract the features
    prediction_feature = get_spectogram(audio, sr, hop_length=hop_length, n_mels=n_mels)
    if mfcc:
        prediction_feature = extract_features(file_path=file_path, n_mfcc=40, sampling_rate=None, num_seg=5, augment=False, split=False, label="Whistling")
    print(prediction_feature[0].shape[1])
    input_shape = (1, prediction_feature[0].shape[0], prediction_feature[0].shape[1], 1)

    visualize_class_activation_map(model_path, prediction_feature[0], 'test.png', 'original.png', input_shape, mfcc=mfcc, sound='rhonchus')
    # Make the input specific to that of a CNN
    pred = prediction_feature[0].reshape(input_shape)

    whistling_labels = ['NO', 'YES']
    rhonchus_labels = ['OK', 'MILD', 'MODERATE', 'SEVERE']

    # Predict the label using model
    predicted_vector = model.predict(pred)
    predicted_class_whistling = predicted_vector[0]
    predicted_class_rhonchus = predicted_vector[1]

    print("Whistling: " + extract_label(whistling_labels, predicted_class_whistling))
    print("Rhonchus: " + extract_label(rhonchus_labels, predicted_class_rhonchus))


if __name__ == '__main__':
    model_path = 'src/saved_models/weights.best.multitask.hdf5'

    model = load_model(model_path)

    audio_path = "src/data/recordings/PV19064/PV19064_270119_R.wav"

    get_prediction(audio_path, model_path, None, mfcc=False)
