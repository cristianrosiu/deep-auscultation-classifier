import librosa
import numpy as np
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt


if __name__ == '__main__':
  

    df = pd.read_csv('labels_int.csv')

    for index, row in df.iterrows():
        path_l = 'recordings/{id}/{name}_L.wav'.format(id=row['seal_id'], name=row['rec_name'])
        path_r = 'recordings/{id}/{name}_R.wav'.format(id=row['seal_id'], name=row['rec_name'])
        
        class_l_w = row['whistling_l']
        class_r_w = row['whistling_r']
        class_l_r = row['rhonchus_l']
        class_r_r = row['rhonchus_r']

        # load audio. Using example from librosa
        signal_l, sr = librosa.load(path_l, offset=1.0, sr=None)
        signal_r, sr = librosa.load(path_r, offset=1.0, sr=None)

        # extract a fixed length window
        start_sample = 0  # starting at beginning
        length_samples = time_steps * hop_length
        window_l = signal_l[start_sample:start_sample + length_samples]
        window_r = signal_r[start_sample:start_sample + length_samples]

        # convert to PNG
        left_lung = spectrogram_image(window_l, sr=sr, hop_length=hop_length, n_mels=n_mels)
        right_lung = spectrogram_image(window_r, sr=sr, hop_length=hop_length, n_mels=n_mels)
        
    print('Finished creating {number} spectograms '.format(number=len(spec_df)))
