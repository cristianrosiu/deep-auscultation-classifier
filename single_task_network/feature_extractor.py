import librosa
import librosa.display
import pandas as pd
import math
import numpy as np

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


def extract_segmented_features(file_path, n_mfcc, num_seg, sampling_rate=None):
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
        data = []
        samples_per_seg = int(SAMPLE_PER_AUDIO / num_seg)
        expected_mfcc_len = math.ceil(samples_per_seg / 512)
        samples_per_seg = int((4000 * librosa.get_duration(audio, sample_rate)) / num_seg)

        for seg in range(num_seg):
            start_seg = samples_per_seg * seg
            end_seg = start_seg + samples_per_seg
            mfccs = librosa.feature.mfcc(y=audio[start_seg:end_seg], sr=sample_rate, n_mfcc=n_mfcc)
            if mfccs.shape[1] == expected_mfcc_len:
                data.append(mfccs)

    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        print(e)
        return None

    return data


def get_expected_len(audio_length, sampling_rate=None):
    sample_per_audio = 4000 * audio_length
    if sampling_rate is not None:
        sample_per_audio = sampling_rate * audio_length
    return math.ceil(sample_per_audio / 512)


def extract_features(file_path, n_mfcc, sampling_rate=None):
    """Returns a list which contains the mfcc features of each segment.
    :param file_path: Path to the audio file
    :param sampling_rate: Sampling rate value used to resample the audio. A default value of None
    will take the original sampling rate of the audio file (in our case is 4000Hz)
    :param n_mfcc: Number of mfcc which will be extracted over a period of time
    :return: A list containing the MFCC features of each segment in the original audio
    """

    # Because the extracted arrays have variable length we need to calculate the maximum possible
    # number of MFCCs that can be extracted from our audio files and then pad with 0 the arrays
    # which have a length < max length.

    expected_mfcc_len = get_expected_len(AUDIO_LENGTH, sampling_rate)
    try:
        audio, sample_rate = librosa.load(file_path, sr=sampling_rate, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = expected_mfcc_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

    return mfccs


def get_features_df(data, sampling_rate=None, label_row='rhonchus', n_mfcc=20):
    features = []
    for index, row in data.iterrows():
        file_name_l = 'recordings/{id}/{recording}_L.wav'.format(id=row['seal_id'], recording=row['rec_name'])
        file_name_r = 'recordings/{id}/{recording}_R.wav'.format(id=row['seal_id'], recording=row['rec_name'])

        class_label_l = row[label_row + '_l']
        class_label_r = row[label_row + '_r']

        features_data_l = extract_features(file_name_l, n_mfcc=n_mfcc, sampling_rate=sampling_rate)
        features_data_r = extract_features(file_name_r, n_mfcc=n_mfcc, sampling_rate=sampling_rate)

        if class_label_l != '?':
            features.append([features_data_l, class_label_l])
        if class_label_r != '?':
            features.append([features_data_r, class_label_r])
    features_df = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(features_df), ' files')

    return features_df


def get_seg_features_df(data, sampling_rate=None, label_row='rhonchus', n_mfcc=20, num_seg=5):
    features = []
    for index, row in data.iterrows():
            file_name_l = 'src/data/recordings/{id}/{recording}_L.wav'.format(id=row['seal_id'], recording=row['rec_name'])
            file_name_r = 'src/data/recordings/{id}/{recording}_R.wav'.format(id=row['seal_id'], recording=row['rec_name'])
            
            class_label_l = row[label_row + '_l']
            class_label_r = row[label_row + '_r']

            features_data_l = extract_segmented_features(file_name_l, n_mfcc=n_mfcc, sampling_rate=sampling_rate,
                                                        num_seg=num_seg)
            features_data_r = extract_segmented_features(file_name_r, n_mfcc=n_mfcc, sampling_rate=sampling_rate,
                                                        num_seg=num_seg)
            if class_label_l != '?':
                for d in features_data_l:
                    features.append([d, class_label_l])
            if class_label_r != '?':
                for d in features_data_r:
                    features.append([d, class_label_r])

    features_df = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(features_df), ' files')

    return features_df


def get_prediction(file_path, model, n_mfcc, sampling_rate=None):
    """
    :param file_path: Path to the audio
    :param model: The model to be used for prediction
    :param n_mfcc: The number of coefficients to be extracted
    :param sampling_rate: Sampling rate
    :return: Predicted label of the audio (i.e: 0 (OK) 1 (MILD) 2(MODERATE) 3(SEVERE)
    """

    num_rows = n_mfcc
    num_columns = get_expected_len(AUDIO_LENGTH, sampling_rate)
    # Always 1 for CNNs
    num_channels = 1

    # Extract the features
    prediction_feature = extract_features(file_path, n_mfcc, sampling_rate)
    # Make the input specific to that of a CNN
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    # Predict the label using model
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = predicted_vector

    return predicted_class[0]
