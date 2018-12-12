# -*- coding: utf-8 -*-
"""
@author: Jason Zhang
@github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob


def data_stat():
    """data statistic"""
    audio_path = './data/esc10/audio/'
    class_list = [os.path.basename(i) for i in glob(audio_path + '*')]
    nums_each_class = [len(glob(audio_path + cl + '/*.ogg')) for cl in class_list]
    rects = plt.bar(range(len(nums_each_class)), nums_each_class)

    index = list(range(len(nums_each_class)))
    plt.title('Numbers of each class for ESC-10 dataset')
    plt.ylim(ymax=60, ymin=0)
    plt.xticks(index, class_list, rotation=45)
    plt.ylabel("numbers")

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_wave(sound_files, sound_names):
    """plot wave"""
    i = 1
    fig = plt.figure(figsize=(20, 64))
    for f, n in zip(sound_files, sound_names):
        y, sr = librosa.load(os.path.join('./data/esc10/audio/', f))
        plt.subplot(10, 1, i)
        librosa.display.waveplot(y, sr, x_axis=None)
        plt.title(n + ' - ' + 'Wave')

        i += 1

    plt.tight_layout(pad=10)
    plt.show()


def plot_spectrum(sound_files, sound_names):
    """plot log power spectrum"""
    i = 1
    fig = plt.figure(figsize=(20, 64))
    for f, n in zip(sound_files, sound_names):
        y, sr = librosa.load(os.path.join('./data/esc10/audio/', f))
        plt.subplot(10, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(y)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, sr=sr, y_axis='log')
        plt.title(n + ' - ' + 'Spectrum')

        i += 1

    plt.tight_layout(pad=10)
    plt.show()

if __name__ == '__main__':
    # 每类样本选取一个demo文件
    sound_files = ['Dog/1-30226-A.ogg', 'Rooster/1-26806-A.ogg', 'Rain/1-17367-A.ogg', 'Sea waves/1-28135-A.ogg',
                   'Crackling fire/1-4211-A.ogg', 'Crying baby/1-22694-A.ogg', 'Sneezing/1-26143-A.ogg',
                   'Clock tick/1-21934-A.ogg', 'Helicopter/1-172649-A.ogg', 'Chainsaw/1-19898-A.ogg']
    # 各类的标签
    sound_names = ['Dog', 'Rooster', 'Rain', 'Sea waves', 'Crackling fire', 'Crying baby', 'Sneezing',
                   'Clock tick', 'Helicopter', 'Chainsaw']

    # 统计数据集中类别数，各类样本的数量，总样本数量等信息
    data_stat()
    # 画出每类样本的波形图，比较各类样本波形之间的差异
    plot_wave(sound_files, sound_names)
    # 画出每类样本的能量谱图，比较各类样本能量谱之间的差异
    plot_spectrum(sound_files, sound_names)


