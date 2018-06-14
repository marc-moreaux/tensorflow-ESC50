#%%
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import itertools


# dataset function
def to_dataset(sounds, labels, batch_size):
    labels = np.array(labels).reshape((-1, 1))
    dataset = tf.data.Dataset.from_generator(
        lambda: itertools.izip_longest(sounds, labels),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([None]),
                       tf.TensorShape(1)))
    return dataset


def get_different_sounds(iterator_next):
    s1, l1 = iterator_next
    s2, l2 = iterator_next
    loop_vars = [s1, l1, s2, l2]

    def cond(s1, l1, s2, l2):
        l1 = tf.gather(l1, 0)
        l2 = tf.gather(l2, 0)
        return tf.equal(l1, l2)

    def body(s1, l1, s2, l2):
        s1, l1 = iterator_next
        s2, l2 = iterator_next
        return s1, l1, s2, l2

    loop = tf.while_loop(cond, body, loop_vars)
    sound1, label1, sound2, label2 = loop
    return sound1, label1, sound2, label2


# General loading functions
def fake_parse():
    from argparse import Namespace
    args = Namespace(save='/home/moreaux-gpu/work/bc_learning_sound/results/esc10_att_7b/',
                     split=[1, ],
                     noiseAugment=False,
                     inputLength=0)
    
    return args


def fix_opt(opt):
    if 'results_' in opt.save:
        opt.save = opt.save.replace('results_', 'results/')
    if not 'noiseAugment' in opt:
        opt.noiseAugment = False
    opt.data = opt.data.replace('moreaux', 'moreaux-gpu')
    return opt


def load_opt(save_path, split):
    '''Load opt stored at <save_path> on split <split>
    '''
    # Load opt
    with open(os.path.join(save_path, 'opt{}.pkl'.format(split)), 'rb') as f:
        opt = pickle.load(f)
        opt = fix_opt(opt)
    
    return opt


# Default data augmentation
def padding(pad):
    def f(sound, label):
        sound = tf.pad(sound, [[pad, pad]], 'constant')
        return sound, label

    return f


def random_crop(size):
    def f(sound, label):
        sound = tf.random_crop(sound, [size])
        return sound, label

    return f


def normalize(factor):
    def f(sound, label):
        sound = sound / factor
        return sound, label

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound, label):
        '''Linear interpolation of a 1D signal
        '''
        # sound = tf.range(10, name='sound', dtype='float32')
        sound_len = tf.shape(sound)[0]
        sound_len_f = tf.cast(sound_len, tf.float32)

        scale = tf.pow(max_scale, random.uniform(-1, 1))
        
        output_len = tf.multiply(sound_len_f, scale)
        output_len = tf.cast(output_len, tf.int32, name='output_len')
        # output_len = sound_len * scale


        ref = tf.range(output_len)
        ref = tf.cast(ref, tf.float32, name='ref')
        ref = ref / scale
        ref1 = tf.floor(ref)
        ref2 = tf.minimum( ref1 + 1, tf.cast(sound_len, tf.float32) - 1)
        r = tf.subtract(ref, ref1, name='r')

        ref1 = tf.cast(ref1, tf.int32, name='plout1')
        ref2 = tf.cast(ref2, tf.int32, name='plout2')
        scaled_sound = tf.gather(sound, ref1) * (1 - r) + tf.gather(sound, ref2) * r
        return scaled_sound, label

    return f


def random_gain(db):
    def f(sound, label):
        sound = sound * tf.pow(10.0, random.uniform(-db, db) / 20.0)
        return sound, label

    return f


def reshape(shape):
    def f(sound, label):
        sound = tf.reshape(sound,shape)
        return sound, label

    return f


def noiseAugment(opt):
    # TODO: transform to tensorflow
    data_path = opt.data
    npz_path = join(data_path, 'noise', 'wav{}.npz'.format(opt.fs // 1000))
    dataset = dict(np.load(npz_path).items())
    train, valid = dataset['train'][0], dataset['valid'][0]
    valid = (valid / np.percentile(train, 95)).clip(-1, 1)
    train = (train / np.percentile(train, 95)).clip(-1, 1)
    valid = tf.constant(valid, dtype=tf.float32)
    train = tf.constant(train, dtype=tf.float32)

    def f(is_train, audio_len):
        ds = train if is_train else valid
        noise = tf.random_crop(ds, audio_len)
        return noise

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    input_length = int(input_length)
    n_crops = int(n_crops)
    def f(sound, label):
        sound_len = tf.shape(sound)[0]
        sound_len_f = tf.cast(sound_len, tf.float32)
        stride = tf.floordiv(sound_len_f - input_length, n_crops - 1)
        stride = tf.cast(stride, tf.int32)
        sounds = [tf.slice(sound, [stride * i,], [input_length,]) for i in range(n_crops)]
        sounds = tf.stack(sounds)
        return sounds, label

    return f


# For BC learning
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def a_weight(fs, n_fft, min_db=-80.0):
    freq = tf.lin_space(0., fs // 2, n_fft // 2 + 1)
    freq_sq = tf.pow(freq, 2)
    a_one = tf.SparseTensor(indices=[[0,]], values=[1.], dense_shape=freq_sq.shape)
    a_one = tf.sparse_tensor_to_dense(a_one)
    freq_sq = freq_sq + a_one
    weight = 2.0 + 20.0 * (2 * log10(12194.) + 2 * log10(freq_sq)
                            - log10(freq_sq + 12194. ** 2)
                            - log10(freq_sq + 20.6 ** 2)
                            - 0.5 * log10(freq_sq + 107.7 ** 2)
                            - 0.5 * log10(freq_sq + 737.9 ** 2))
    weight = tf.maximum(weight, min_db)
    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2
    sound_len = int(sound.shape[0])

    gain = []
    for i in xrange(0, sound_len - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = tf.spectral.rfft(tf.contrib.signal.hann_window(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = tf.pow(tf.abs(spec), 2.)
            a_weighted_spec = power_spec * tf.pow(10., a_weight(fs, n_fft) / 10)
            g = tf.reduce_sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = tf.stack(gain)
    gain = tf.maximum(gain, np.power(10., min_db / 10.))
    gain_db = 10 * log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = tf.reduce_max(compute_gain(sound1, fs))  # Decibel
    gain2 = tf.reduce_max(compute_gain(sound2, fs))
    t = 1. / (1. + tf.pow(10., (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / tf.sqrt(t ** 2 + (1 - t) ** 2))

    return sound



#%%
def kl_divergence(logits, labels):
    zero = tf.constant(0, dtype=tf.float32)
    labels = tf.cast(labels, tf.float32)
    where = tf.not_equal(labels, zero)
    sound_non_zero = tf.boolean_mask(labels, where)

    entropy = - tf.reduce_sum(sound_non_zero * tf.log(sound_non_zero))
    crossEntropy = labels * tf.nn.log_softmax(logits)
    crossEntropy = - tf.reduce_sum(crossEntropy)
    kl_dvg = (crossEntropy - entropy) / tf.cast(logits.shape[0], tf.float32)

    return kl_dvg


# Convert time representation



# For testing purposes
# logits = np.array([[.1,.9,0],[.8,.2,0]])
# labels = np.array([[0,1,0],[1,0,0]])

# logits = tf.constant(logits, dtype=tf.float32)
# labels = tf.constant(labels, dtype=tf.int32)

# print(logits[0])
# with tf.Session() as sess:
#     op = kl_divergence(logits, labels)
#     results, = sess.run([op])
#     print (results)
