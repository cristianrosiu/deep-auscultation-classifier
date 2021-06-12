import librosa
import librosa.display
from librosa.feature.spectral import mfcc
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from src.util.data_augmenter import augment_sample
from sklearn import preprocessing

"""
This script allows the user to extract the mel-cepstral coefficients array from a 
given audio file. It loads the audio file, gets the MFCC array and transforms it such 
that it is valid input for a CNN.

This tool accepts the file path to an audio file, the desired sampling rate and the number of MFCCs.
It returns the coefficients array of the given audio file.

I have also took the time to create a main() where you can see a example of how this tool can be used.

This file can also be imported as a module and contains the following
functions:
    * extract_features - Returns a list which contains the mfcc features of an audio file.
    * get_prediction   - Extracts the features of an audio file and inputs them into the model. 
                         Returns the predicted label

"""

# In our case all audio files have the same length of 30
AUDIO_LENGTH = 30


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def get_mfcc(features):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(features, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def get_expected_len(audio_length, sampling_rate=None):
    sample_per_audio = 4000 * audio_length
    if sampling_rate is not None:
        sample_per_audio = sampling_rate * audio_length
    return math.ceil(sample_per_audio / 512)


def extract_features(file_path, n_mfcc, num_seg, sampling_rate=None, split=False, pcen=False):
    """Returns a list which contains the mfcc features of each segment.
    :param split:
    :param pcen:
    :param file_path: Path to the audio file
    :param sampling_rate: Sampling rate value used to resample the audio. A default value of None
    will take the original sampling rate of the audio file (in our case is 4000Hz)
    :param n_mfcc: Number of mfcc which will be extracted over a period of time
    :param num_seg: Number of segments in which the audio will divide
    :return: A list containing the MFCC features of each segment in the original audio
    """

    SAMPLE_PER_AUDIO = 4000 * AUDIO_LENGTH
    if sampling_rate is not None:
        SAMPLE_PER_AUDIO = sampling_rate * AUDIO_LENGTH

    try:
        audio, sample_rate = librosa.load(file_path, sr=sampling_rate, res_type='kaiser_best')
        data = []
        # Generate features
        if pcen is True:
            S = librosa.feature.melspectrogram(audio, sr=sample_rate, power=1)
            pcen_S = librosa.pcen(S * (2 ** 31))
            pad_width = 235 - pcen_S.shape[1]
            img = np.pad(pcen_S, pad_width=((0, 0), (0, pad_width)), mode='constant')
            data.append(img)
        else:
            if split:
                samples_per_seg = int(SAMPLE_PER_AUDIO / num_seg)
                expected_mfcc_len = math.ceil(samples_per_seg / 512)
                samples_per_seg = int((4000 * librosa.get_duration(audio, sample_rate)) / num_seg)
                for seg in range(num_seg):
                    start_seg = samples_per_seg * seg
                    end_seg = start_seg + samples_per_seg
                    coefficients = librosa.feature.mfcc(y=audio[start_seg:end_seg], sr=sample_rate, n_mfcc=n_mfcc)
                    pad_width = expected_mfcc_len - coefficients.shape[1]
                    coefficients = np.pad(coefficients, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    data.append(coefficients)
            else:
                expected_mfcc_len = get_expected_len(AUDIO_LENGTH, sampling_rate)
                # Generate features
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
                # Pad with 0's in case the audio doesn't have length 30
                pad_width = expected_mfcc_len - mfccs.shape[1]
                img = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                data.append(img)

    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        print(e)
        return None

    return data


def get_features_df(data, sampling_rate=None, n_mfcc=20, num_seg=5, split=False, pcen=False, augment=False):
    features = []
    for index, row in data.iterrows():
        if augment is False:
            file_name_l = 'data/recordings/{id}/{recording}_L.wav'.format(id=row['seal_id'], recording=row['rec_name'])
            file_name_r = 'data/recordings/{id}/{recording}_R.wav'.format(id=row['seal_id'], recording=row['rec_name'])

            features_data_l = extract_features(file_name_l, n_mfcc=n_mfcc, sampling_rate=sampling_rate, num_seg=num_seg,
                                               split=split, pcen=pcen)
            features_data_r = extract_features(file_name_r, n_mfcc=n_mfcc, sampling_rate=sampling_rate, num_seg=num_seg,
                                               split=split, pcen=pcen)

            for left_lung in features_data_l:
                features.append([left_lung, [row['whistling_l'], row['rhonchus_l']]])
            for right_lung in features_data_r:
                features.append([right_lung, [row['whistling_r'], row['rhonchus_r']]])
        else:
            file_name = 'data/augmented_recordings/{name}.wav'.format(name=row['rec_name'])
            features_data = extract_features(file_name, n_mfcc=n_mfcc, sampling_rate=sampling_rate, num_seg=num_seg,
                                             split=split, pcen=pcen)
            row_idx = row['row_id']

            if '_r' in row_idx:
                for lung in features_data:
                    features.append([lung, [row['whistling_r'], row['rhonchus_r']]])
            elif '_l' in row_idx:
                for lung in features_data:
                    features.append([lung, [row['whistling_l'], row['rhonchus_l']]])

    features_df = pd.DataFrame(features, columns=['feature', 'class_labels'])

    print('Finished feature extraction from ', len(features_df), ' files')

    return features_df
