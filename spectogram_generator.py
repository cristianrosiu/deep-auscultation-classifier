import librosa
import numpy as np
import pandas as pd


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
    pad_width = 128 - mels.shape[1]
    mels = np.pad(mels, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    return img


def get_spec_df(df, label_row):
    # settings
    hop_length = 512  # number of samples per time-step in spectrogram
    n_mels = 128  # number of bins in spectrogram. Height of image
    time_steps = 127  # number of time-steps. Width of image

    features = []
    for index, row in df.iterrows():
        path_l = 'recordings/{id}/{name}_l.wav'.format(id=row['seal_id'], name=row['rec_name'])
        path_r = 'recordings/{id}/{name}_r.wav'.format(id=row['seal_id'], name=row['rec_name'])

        class_l = row[label_row + '_l']
        class_r = row[label_row + '_r']

        # load audio. Using example from librosa
        signal_l, sr = librosa.load(path_l, offset=1.0, duration=30.0, sr=None)
        signal_r, sr = librosa.load(path_r, offset=1.0, duration=30.0, sr=None)

        # extract a fixed length window
        start_sample = 0  # starting at beginning
        length_samples = time_steps * hop_length
        window_l = signal_l[start_sample:start_sample + length_samples]
        window_r = signal_r[start_sample:start_sample + length_samples]

        # convert to PNG
        img_l = spectrogram_image(window_l, sr=sr, hop_length=hop_length, n_mels=n_mels)
        img_r = spectrogram_image(window_r, sr=sr, hop_length=hop_length, n_mels=n_mels)

        if class_l != '?':
            features.append([img_l, class_l])
        if class_r != '?':
            features.append([img_r, class_r])

    features_df = pd.DataFrame(features, columns=['feature', 'class_label'])
    print('Finished feature extraction from ', len(features_df), ' files')
    return features_df
