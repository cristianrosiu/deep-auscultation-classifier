import pandas as pd
import numpy as np

from features.feature_extractor import extract_features


def generate_data(path_to_metadata='../data/metadata.csv', path_to_data='../data/recordings', pcen=False):
    """
    This function generates the data necessary for our model to learn.

    :param path_to_metadata: Path to the .csv containing information about our dataset
    :param path_to_data: Path to the actual audio recordings
    :param pcen: Boolean that checks whether or not the script should extract PCEN features or not
    :return: Generated data and the ground truth labels of each task
    """

    metadata = pd.read_csv(path_to_metadata)
    metadata = metadata.sample(frac=1).reset_index(drop=True)

    features_df = extract_features(metadata, path_to_data, sr=None, pcen=pcen)

    X = np.array(features_df.features.to_list())
    y = np.array(features_df.whistling.to_list())
    y1 = np.array(features_df.rhonchus.to_list())
    y2 = np.array(features_df.survival.to_list())

    return X, y, y1, y2
