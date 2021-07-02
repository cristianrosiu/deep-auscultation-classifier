import librosa
import librosa.display
from librosa.feature.spectral import mfcc
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

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

SAMPLE_RATE = 4000
TRACK_DURATION = 23  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def pcen2bw(pcen):
    img = scale_minmax(pcen, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    plt.imshow(img)
    return img


def extract_mfcc(y, sr=None, n_mfcc=13, split=False, num_seg=5, n_fft=2048, hop_length=256):
    """Returns a list which contains the mfcc features of each segment.
    :param y: Audio signal
    :param sr: Sampling rate
    :param n_mfcc: Number of MFCC features to be extracted
    :param split: Bool that tells the script to either split or not split the audio signal into segments
    :param num_seg: Number of segments in which the audio will divide
    :param n_fft: Window size
    :param hop_length: Number of samples between each successive FFT window
    :return: A list containing the MFCC features of each segment in the original audio
    """

    samples_per_seg = int(SAMPLES_PER_TRACK / num_seg)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_seg / hop_length)
    try:
        data = []
        if split:
            for seg in range(num_seg):
                start_seg = samples_per_seg * seg
                end_seg = start_seg + samples_per_seg
                mfccs = librosa.feature.mfcc(y=y[start_seg:end_seg], sr=sr,
                                             n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfccs = mfccs.T
                if len(mfccs) == num_mfcc_vectors_per_segment:
                    data.append(mfccs.tolist())
        else:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            data.append(mfccs)
        return data
    except Exception as e:
        print("Error encountered while parsing files")
        print(e)
        return


def extract_pcen(y, sr=4000):
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, power=1)
        pcen_S = librosa.pcen(S * (2 ** 31), sr=sr)
        return [pcen_S]
    except Exception as e:
        print("Error encountered while parsing file")
        print(e)
        return


def extract_features(metadata, path, sr=None, pcen=False):
    features = []

    for index, row in metadata.iterrows():
        left_lung = path + '/{id}/{recording}_L.wav'.format(id=row['seal_id'], recording=row['rec_name'])
        right_lung = path + '/{id}/{recording}_R.wav'.format(id=row['seal_id'], recording=row['rec_name'])

        left_audio, sr = librosa.load(left_lung, sr=sr, duration=23)
        right_audio, sr = librosa.load(right_lung, sr=sr, duration=23)

        left_feautres = extract_mfcc(left_audio, sr, 40, split=True, num_seg=5)
        right_features = extract_mfcc(right_audio, sr, 40, split=True, num_seg=5)
        if pcen:
            left_feautres = extract_pcen(left_audio, sr)
            right_features = extract_pcen(right_audio, sr)

        for left_lung in left_feautres:
            features.append([left_lung, row['whistling_l'], row['rhonchus_l'], row['survival']])
        for right_lung in right_features:
            features.append([right_lung, row['whistling_r'], row['rhonchus_r'], row['survival']])

    features_df = pd.DataFrame(features, columns=['features', 'whistling', 'rhonchus', 'survival'])

    return features_df
