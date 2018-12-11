import numpy as np
import librosa
import random


def extract_logmel(y, sr, size=3):
    """
    extract log mel spectrogram
    :param y: the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :return: log-mel spectrogram feature
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    start = random.randint(0, len(y) - size * sr)
    y = y[start: start + size * sr]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec


def extract_mfcc(y, sr, size=3):
    """
    extract MFCC feature
    :param y: np.ndarray [shape=(n,)], real-valued the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :return: MFCC feature
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    start = random.randint(0, len(y) - size * sr)
    y = y[start: start + size * sr]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    mfcc_comb = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta], axis=0)

    return mfcc_comb


if __name__ == '__main__':
    # a demo sample
    y, sr = librosa.load('./data/esc10/audio/Chainsaw/1-19898-A.ogg')
    feat = extract_mfcc(y, sr, size=3)
