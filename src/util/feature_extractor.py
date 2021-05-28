import librosa
import librosa.display
from librosa.feature.spectral import mfcc
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from data_augmenter import augment_sample


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

def get_spectogram(y, sr, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
    pad_width = 235 - mels.shape[1]
    mels = np.pad(mels, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    #img = 255 - img  # invert. make black==more energy
 
    return(img)

def get_expected_len(audio_length, sampling_rate=None):
    sample_per_audio = 4000 * audio_length
    if sampling_rate is not None:
        sample_per_audio = sampling_rate * audio_length
    return math.ceil(sample_per_audio / 512)


def extract_features(file_path, n_mfcc, num_seg, label, sampling_rate=None, augment=False, split=False):
    """Returns a list which contains the mfcc features of each segment.
    :param file_path: Path to the audio file
    :param sampling_rate: Sampling rate value used to resample the audio. A default value of None
    will take the original sampling rate of the audio file (in our case is 4000Hz)
    :param n_mfcc: Number of mfcc which will be extracted over a period of time
    :param num_seg: Number of segments in which the audio will divide
    :return: A list containing the MFCC features of each segment in the original audio
    """

    AUDIO_LENGTH = 30
    SAMPLE_PER_AUDIO = 4000 * AUDIO_LENGTH
    if sampling_rate is not None:
        SAMPLE_PER_AUDIO = sampling_rate * AUDIO_LENGTH

    try:
        audio, sample_rate = librosa.load(file_path, sr=sampling_rate, res_type='kaiser_best')
        augmented_audio = augment_sample(y=audio, sample_rate=sample_rate)
        samples_per_seg = int(SAMPLE_PER_AUDIO / num_seg)
        expected_mfcc_len = math.ceil(samples_per_seg / 512)
        samples_per_seg = int((4000 * librosa.get_duration(audio, sample_rate)) / num_seg)

        data = []

        if split:
            for seg in range(num_seg):
                start_seg = samples_per_seg * seg
                end_seg = start_seg + samples_per_seg
                mfccs = librosa.feature.mfcc(y=audio[start_seg:end_seg], sr=sample_rate, n_mfcc=n_mfcc)
                augmented_mfccs = librosa.feature.mfcc(y=augmented_audio[start_seg:end_seg], sr=sample_rate, n_mfcc=n_mfcc)
                if mfccs.shape[1] == expected_mfcc_len:
                    data.append(mfccs)
                if label in [2, 3] and augmented_mfccs.shape[1] == expected_mfcc_len and augment == True:
                    data.append(augmented_mfccs)
        else:
            expected_mfcc_len = get_expected_len(AUDIO_LENGTH, sampling_rate)
            # Load audio file
            audio, sample_rate = librosa.load(file_path, sr=sampling_rate, res_type='kaiser_fast')
            # Generate features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            pad_width = expected_mfcc_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            data.append(mfccs)
            if augment and label in [2, 3]:
                augmented_audio = augment_sample(y=audio, sample_rate=sample_rate)
                augmented_mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=n_mfcc)
                pad_width_augmented = expected_mfcc_len - augmented_mfccs.shape[1]
                augmented_mfccs =  np.pad(augmented_mfccs, pad_width=((0, 0), (0, pad_width_augmented)), mode='constant')
                data.append(augmented_mfccs)
      

    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        print(e)
        return None

    return data


def get_features_df(data, sampling_rate=None, n_mfcc=20, num_seg=5, spectogram=False):
    features = []
    for index, row in data.iterrows():
        file_name_l = 'src/data/recordings/{id}/{recording}_L.wav'.format(id=row['seal_id'], recording=row['rec_name'])
        file_name_r = 'src/data/recordings/{id}/{recording}_R.wav'.format(id=row['seal_id'], recording=row['rec_name'])

        features_data_l = None
        features_data_r = None
        if spectogram:
            # settings
            hop_length = 512  # number of samples per time-step in spectrogram
            n_mels = 128  # number of bins in spectrogram. Height of image

            # load audio. Using example from librosa
            signal_l, sr = librosa.load(file_name_l, sr=None)
            signal_r, sr = librosa.load(file_name_r, sr=None)

            # extract a fixed length window
            #start_sample = 0  # starting at beginning
            #length_samples = time_steps * hop_length
            #window_l = signal_l[start_sample:start_sample + length_samples]
            #window_r = signal_r[start_sample:start_sample + length_samples]
            features_data_l = get_spectogram(signal_l, sr, hop_length=hop_length, n_mels=n_mels)
            features_data_r = get_spectogram(signal_r, sr, hop_length=hop_length, n_mels=n_mels)
        else:
            features_data_l = extract_features(file_name_l, n_mfcc=n_mfcc, sampling_rate=sampling_rate, num_seg=num_seg,label=row['rhonchus_l'], augment=True, split=False)
            features_data_r = extract_features(file_name_r, n_mfcc=n_mfcc, sampling_rate=sampling_rate, num_seg=num_seg, label=row['rhonchus_r'], augment=True, split=False)
        
       
        if spectogram:
            features.append([features_data_l, [row['whistling_l'], row['rhonchus_l']]])
            features.append([features_data_r, [row['whistling_r'], row['rhonchus_r']]])
        else:
            for left_lung in features_data_l:
                features.append([left_lung, [row['whistling_l'], row['rhonchus_l']]])
            for right_lung in features_data_r:
                features.append([right_lung, [row['whistling_r'], row['rhonchus_r']]])
        

    features_df = pd.DataFrame(features, columns=['feature', 'class_labels'])

    print('Finished feature extraction from ', len(features_df), ' files')

    return features_df

