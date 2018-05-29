""" Convolutional Neural Network.
"""
#%%
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import dataset
reload(dataset)
import utils


# Model function
def conv_net(features, labels, mode, params):
    '''
    features: a dict of tensors
    labels:
    mode: one of tf.estimator.ModeKeys
    params: a dict of parameters
    '''
    n_classes = params['n_classes']
    is_training = False
    if mode is tf.estimator.ModeKeys.TRAIN:
        is_training = True
    
    with tf.variable_scope('ConvNet'):
        x = features #features['sound']
        # x = tf.Print(x, [x], 'my x bitch')
        x = tf.layers.conv1d(x, 32, 5, activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.max_pooling1d(x, 2, 2)

        x = tf.layers.conv1d(x, 64, 3, activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.max_pooling1d(x, 2, 2)

        x = tf.layers.conv1d(x, n_classes, 3, activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.reduce_mean(x, [1])

        logits = x

    with tf.variable_scope('predictions'):
        predicted_classes = tf.argmax(logits, 1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.variable_scope('evaluate'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
    
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
    
    with tf.variable_scope('train'):
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    classifier = tf.estimator.Estimator(
        model_fn=conv_net,
        params={
            'n_classes':10,
        }
    )

    classifier.train(
        input_fn=dataset.train_input_function,
        steps=1000,
    )
    print('3')

    eval_result = classifier.evaluate(
        input_fn=dataset.val_input_function
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


main(None)
print("hello")
