""" Convolutional Neural Network.
"""
#%%
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import dataset
#reload(dataset)
import utils

use_BC=True
# Model function
def conv_net(features, labels, mode, params):
    '''
    features: a dict of tensors
    labels:
    mode: one of tf.estimator.ModeKeys
    params: a dict of parameters
    use_BC: use kl dvg or not
    '''
    n_classes = params['n_classes']
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    print (is_training)
    
    x = features #features['sound']
    x = tf.Print(x, [labels, x], 'lbl et x')
    for _ in range(12):
        x = tf.layers.conv1d(x, 64, 3, strides=2, activation=tf.nn.relu)
#        x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.conv1d(x, n_classes, 3, strides=2)
    x = tf.reduce_mean(x, [1])

    logits = x
    logits = tf.Print(logits, [logits], 'logits')

    predicted_classes = tf.argmax(logits, -1)

    # Predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': tf.nn.softmax(logits),
            'class_ids': predicted_classes,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    labels = tf.Print(labels, [labels], 'lbl')
    predicted_classes = tf.Print(predicted_classes, [predicted_classes], 'pred')

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                    predictions=predicted_classes,
                                    name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # Eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Train mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    loss = tf.Print(loss, [loss, predicted_classes])
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    tf.reset_default_graph()
    classifier = tf.estimator.Estimator(
        model_fn=conv_net,
        params={
            'n_classes':10,
        }
    )

    for _ in range(5):
        classifier.train(input_fn=lambda : dataset.dataset_input_fn(True, 8))
        eval_result = classifier.evaluate(input_fn=lambda : dataset.dataset_input_fn(False, 8))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
        print(eval_result)
        # predictions = list(classifier.predict(input_fn=lambda : dataset.dataset_input_fn(True, 8)))
        # print(predictions)



main(None)
print("hello")
