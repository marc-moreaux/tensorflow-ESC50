""" Convolutional Neural Network.
"""
#%%
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import dataset
#reload(dataset)
import utils


def your_model_fn(input_tensor, is_training, n_classes):
    x = input_tensor
    for i in range(12):
        x = tf.layers.conv1d(x, 64, 3, strides=2)
        x = tf.layers.batch_normalization(x, training=is_training, axis=-1)
        tf.summary.histogram("bn/" + str(i), x)
        x = tf.nn.relu(x)

 
    x = tf.layers.conv1d(x, n_classes, 3, strides=2)
    x = tf.reduce_mean(x, [1])
 
    logits = x
    logits = tf.Print(logits, [logits], 'logits')
    return logits


use_BC=True
# Model function
def model_fn(features, labels, mode, params):
    is_training = bool(mode == tf.estimator.ModeKeys.TRAIN)
    
    n_classes = params['n_classes']
    input_tensor = features
    logits = your_model_fn(input_tensor, is_training=is_training, n_classes=n_classes)
    probs = tf.nn.softmax(logits, name="output_score")
    predictions = tf.argmax(probs, axis=-1, name="output_label")
    # provide a tf.estimator spec for PREDICT
    predictions_dict = {"score": probs,
                        "label": predictions}
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_output = tf.estimator.export.PredictOutput(predictions_dict)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions_dict,
                                          export_outputs={
                                              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predictions_output
                                          })
    # calculate loss
    # onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), n_classes)
    # logits = tf.reshape(logits, [64, 1, 10])
    # loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = 0.001
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tensors_to_log = {'accuracy': accuracy[0],
                          'logits': logits,
                          'label': labels}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook], 
                                          eval_metric_ops={"accuracy": accuracy})
    else:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)


def main():
    tf.reset_default_graph()
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'n_classes':10,
        }
    )

    for _ in range(50):
        classifier.train(input_fn=lambda : dataset.dataset_input_fn(True, 64))
        eval_result = classifier.evaluate(input_fn=lambda : dataset.dataset_input_fn(True, 64))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
        print(eval_result)
        # predictions = list(classifier.predict(input_fn=lambda : dataset.dataset_input_fn(True, 8)))
        # print(predictions)



main()
print("hello")
