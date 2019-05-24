import matplotlib.pyplot as plt
import librosa
import os


def aug_pitch_shift(input_audio):
    dirpath = os.path.dirname(input_audio)
    topath = dirpath.replace('audio', 'aug_audio')
    if not os.path.isdir(topath):   # 判断目标路径是否存在，不存在则创建
        os.mkdir(topath)
    y, sr = librosa.load(input_audio, sr=44100)
    for step in [-3, -2, -1, 1, 2, 3]:
        y_shift = librosa.effects.pitch_shift(y, sr, n_steps=step)   # 使用PS生成新数据
        librosa.output.write_wav(os.path.join(topath,    # 数据导出为文件
            os.path.basename(input_audio).replace('.ogg', '_ps{}.ogg'.format(step))), y_shift, sr)


def aug_time_stretch(input_audio):
    dirpath = os.path.dirname(input_audio)
    topath = dirpath.replace('audio', 'aug_audio')
    if not os.path.isdir(topath):   # 判断目标路径是否存在，不存在则创建
        os.mkdir(topath)
    y, sr = librosa.load(input_audio, sr=44100)
    for rate in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
        y_shift = librosa.effects.time_stretch(y, rate=rate)   # 使用TS生成新数据
        librosa.output.write_wav(os.path.join(topath,    # 数据导出为文件
            os.path.basename(input_audio).replace('.ogg', '_ts{}.ogg'.format(rate))), y_shift, sr)


def demo_plot():
    audio = './data/esc10/audio/Dog/1-30226-A.ogg'
    y, sr = librosa.load(audio, sr=44100)
    y_ps = librosa.effects.pitch_shift(y, sr, n_steps=6)   # n_steps控制音调变化尺度
    y_ts = librosa.effects.time_stretch(y, rate=1.2)   # rate控制时间维度的变换尺度
    plt.subplot(311)
    plt.plot(y)
    plt.title('Original waveform')
    plt.axis([0, 200000, -0.4, 0.4])
    # plt.axis([88000, 94000, -0.4, 0.4])
    plt.subplot(312)
    plt.plot(y_ts)
    plt.title('Time Stretch transformed waveform')
    plt.axis([0, 200000, -0.4, 0.4])
    plt.subplot(313)
    plt.plot(y_ps)
    plt.title('Pitch Shift transformed waveform')
    plt.axis([0, 200000, -0.4, 0.4])
    # plt.axis([88000, 94000, -0.4, 0.4])
    plt.tight_layout()
    plt.show()

demo_plot()
