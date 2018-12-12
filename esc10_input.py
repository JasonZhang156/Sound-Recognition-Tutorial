# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

import numpy as np

RANDOM_SEED = 20181212


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=int)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def get_data(test_fold, feat):
    """load feature for train and test"""
    # load feature
    data = np.load('./data/esc10/feature/esc10_{}_fold{}.npz'.format(feat, test_fold))
    train_x = np.expand_dims(data['train_x'], axis=-1)
    train_y = data['train_y']
    test_x = np.expand_dims(data['test_x'], axis=-1)
    test_y = data['test_y']

    # one-hot encode
    train_y = dense_to_one_hot(train_y, 10)
    test_y = dense_to_one_hot(test_y, 10)

    # z-score normalization
    mean = np.mean(train_x)
    std = np.std(train_x)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    # shuffle
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_x)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_y)

    print('Audio Feature: ', feat)
    print('Training Set Shape: ', train_x.shape)
    print('Test Set Shape: ', test_x.shape)

    return train_x, train_y, test_x, test_y