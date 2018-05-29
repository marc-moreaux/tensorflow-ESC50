#%%
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import os
import itertools
import numpy as np
import utils
import utils as U
reload(utils)
U = utils


args = U.fake_parse()
opt = U.load_opt(args.save, 1)
dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)))
mix = opt.BC

# Split train and val
split = 1
train_sounds = []
train_labels = []
val_sounds = []
val_labels = []
for i in range(1, opt.nFolds + 1):
    sounds = dataset['fold{}'.format(i)].item()['sounds']
    labels = dataset['fold{}'.format(i)].item()['labels']
    if i == split:
        val_sounds.extend(sounds)
        val_labels.extend(labels)
    else:
        train_sounds.extend(sounds)
        train_labels.extend(labels)


def dataset_input_fn(sounds, labels, is_train):
    labels = np.array(labels).reshape((-1, 1))
    print('plout')
    print(len(sounds), len(labels))
    dataset = tf.data.Dataset.from_generator(
        lambda: itertools.izip_longest(sounds, labels),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([None]),
                       tf.TensorShape(1)))

    if is_train:
        if opt.strongAugment:
            dataset = dataset.map(U.random_scale(1.25))
        dataset = dataset.map(U.padding(opt.inputLength // 2))
        dataset = dataset.map(U.random_crop(opt.inputLength))
        dataset = dataset.map(U.normalize(float(2 ** 16 / 2)))

    else:
        if not opt.longAudio:
            dataset = dataset.map(U.padding(opt.inputLength // 2))
        dataset = dataset.map(U.normalize(float(2 ** 16 / 2)))
        dataset = dataset.map(U.multi_crop(opt.inputLength, opt.nCrops))

    dataset = dataset.map(U.reshape([1, -1, 1]))
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

train_input_function = lambda: dataset_input_fn(train_sounds, train_labels, True)
val_input_function = lambda: dataset_input_fn(val_sounds, val_labels, False)

# train TF dataset
train_dataset = U.to_dataset(train_sounds, train_labels, opt.batchSize)
if opt.strongAugment:
    train_dataset = train_dataset.map(U.random_scale(1.25))
train_dataset = train_dataset.map(U.padding(opt.inputLength // 2))
train_dataset = train_dataset.map(U.random_crop(opt.inputLength))
train_dataset = train_dataset.map(U.normalize(float(2 ** 16 / 2)))
# train_dataset = train_dataset.repeat()
train_iterator = train_dataset.make_initializable_iterator()


# valid TF dataset
val_dataset = U.to_dataset(val_sounds, val_labels, opt.batchSize)
if not opt.longAudio:
    val_dataset = val_dataset.map(U.padding(opt.inputLength // 2))
val_dataset = val_dataset.map(U.normalize(float(2 ** 16 / 2)))
val_dataset = val_dataset.map(U.multi_crop(opt.inputLength, opt.nCrops))
val_iterator = val_dataset.make_initializable_iterator()


def get_example(iterator, train):
    '''Get a training or testing sample
    '''
    if mix:  # Training phase of BC learning
        sound1, label1, sound2, label2 = U.get_different_sounds(iterator)
        ratio = np.array(np.random.random())
        sound = U.mix(sound1, sound2, ratio, opt.fs)
        eye = tf.eye(opt.nClasses)
        label = tf.gather(eye, label1) * ratio
        label += tf.gather(eye, label2) * (1 - ratio)

    elif not train and opt.longAudio > 0:  # Mix two audio on long frame (for testing)
        sound1, label1, sound2, label2 = U.get_different_sounds(iterator)
        raise NotImplementedError()

    else:  # Training phase of standard learning or testing phase
        sound, label = iterator.get_next()

    if opt.noiseAugment:
        sound_len = tf.shape(sound)[0]
        sound = sound + 0.01 * next_noise(is_train, sound_len)

    if train and opt.strongAugment:
        sound, label = U.random_gain(6)(sound, label)

    return sound, label


train_sound, train_label = get_example(train_iterator, train=True)
val_sound, val_label = get_example(val_iterator, train=False)


def main():
    # For testing purposes
    with tf.Session() as sess:
        import matplotlib.pyplot as plt
        train_iterator = train_input_function()
        sess.run(train_iterator.initializer)
        
        for _ in range(10000):
            results = sess.run([train_iterator])
            a, b = results
            print(results)
            print(a.shape)
            plt.plot(a.T)
            plt.show()


if __name__ == '__main__':
    main()


