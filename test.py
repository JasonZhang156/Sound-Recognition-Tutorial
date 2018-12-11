from keras.models import load_model
import tensorflow as tf
import esc10_input
import numpy as np
import os


def use_gpu():
    """Configuration for GPU"""
    from keras.backend.tensorflow_backend import set_session
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.InteractiveSession(config=config))


def CNN_test(test_fold, feat):
    """
    Test model using test set
    :param test_fold: test fold of 5-fold cross validation
    :param feat: which feature to use
    """
    # 读取测试数据
    _, _, test_features, test_labels = esc10_input.get_data(test_fold, feat)

    # 导入训练好的模型
    model = load_model('./saved_model/cnn_{}_fold{}.h5'.format(feat, test_fold))

    # 输出训练好的模型在测试集上的表现
    score = model.evaluate(test_features, test_labels)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return score[1]


if __name__ == '__main__':
    use_gpu()
    dict_acc = {}
    print('### [Start] Test model for ESC10 dataset #####')
    for fold in [1, 2, 3, 4, 5]:
        print("## Start test fold{} model #####".format(fold))
        acc = CNN_train(fold, 'logmel')
        dict_acc['fold{}'.format(fold)] = acc
        print("## Finish test fold{} model #####".format(fold))
    dict_acc['mean'] = np.mean(list(dict_acc.values()))
    print(dict_acc)
    print('### [Finish] Test model finished for ESC10 dataset #####')
