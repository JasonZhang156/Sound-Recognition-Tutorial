import numpy as np
from glob import glob
import os
import librosa
import feature_extraction as fe
import warnings
warnings.filterwarnings('ignore')


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_esc10_feat():
    # 5-fold cross validation settings
    cv_index = np.load('cvindex.npz')
    TR = cv_index['TR']  # training fold
    TE = cv_index['TE']  # test fold

    # 5 fold extracting
    for fold in range(1, 6):
        print('Extracting training set and test set for fold {}'.format(fold))

        train_feats = []
        train_labels = []
        test_feats = []
        test_labels = []

        class_list = np.sort(glob('./data/esc10/audio/*'))
        for index, classpath in enumerate(class_list):
            print('Total numbers of audio classes: ', len(class_list))
            audio_list = np.sort(glob(os.path.join(classpath, '*.ogg')))
            # for training
            train_audio = audio_list[TR[fold-1]]
            for i in range(len(train_audio)):
                print('Processing sound class: ', os.path.basename(classpath), index+1, '/', len(class_list),
                      ' --- training set: ', i+1, '/', len(train_audio))
                y, fs = librosa.load(train_audio[i], sr=22050)
                # feat = fe.extract_mfcc(y, fs, 3)
                feat = fe.extract_logmel(y, fs, 3)
                train_feats.append(feat)
                train_labels.append(index)

            # for test
            test_audio = audio_list[TE[fold - 1]]
            for j in range(len(test_audio)):
                print('Processing sound class: ', os.path.basename(classpath), index+1, '/', len(class_list),
                      ' --- test set: ', j+1, '/', len(test_audio))
                y, fs = librosa.load(test_audio[j], sr=22050)
                # feat = fe.extract_mfcc(y, fs, 3)
                feat = fe.extract_logmel(y, fs, 3)
                test_feats.append(feat)
                test_labels.append(index)

        train_feats = np.array(train_feats)
        train_labels = np.array(train_labels)
        test_feats = np.array(test_feats)
        test_labels = np.array(test_labels)

        # np.savez('./data/esc10/feature/esc10_mfcc_fold{}.npz'.format(fold),
        #          train_x=train_feats, train_y=train_labels, test_x=test_feats, test_y=test_labels)
        np.savez('./data/esc10/feature/esc10_logmel_fold{}.npz'.format(fold),
                 train_x=train_feats, train_y=train_labels, test_x=test_feats, test_y=test_labels)


if __name__ == '__main__':
    extract_esc10_feat()