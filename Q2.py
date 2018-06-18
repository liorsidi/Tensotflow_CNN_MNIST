from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.monitors import replace_monitors_with_hooks, ValidationMonitor

tf.logging.set_verbosity(tf.logging.INFO)

# Train a Simple CNN network on Mnist

def cnn_model_fn(features, labels, mode,params):
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(features["x"], params['input_layer_dim'] )

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=params['conv_dim'],
        padding="same",
        activation=tf.nn.relu)

    # Q3
    # Norm Layer #1
    # norm1 = tf.layers.batch_normalization(inputs=conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=params['pool_dim'], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=params['conv_dim'],
        padding="same",
        activation=tf.nn.relu)

    # Q3
    # Norm Layer #2
    # norm2 = tf.layers.batch_normalization(
    #     inputs=conv2)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=params['pool_dim'], strides=2)

    # Dense Layer 1
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=params['dense_units'], activation=tf.nn.relu)

    # Q3
    # Dropout Layer
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=params['dropout_rate'])

    # Dense Layer 2
    dense2 = tf.layers.dense(inputs=dense, units=params['dense_units'], activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense2, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "loss" : tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = params['learning_rate']

        #Q5b
        # if params['optimizer'] == 'adam':
        #     optimizer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate)
        # else:
        # Q5a
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        #                                            params['lr_reduce_every_n'], params['lr_reduce_by'], staircase=True)
        #Q2 3
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate)
         #Q1e
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        tensors_to_log = {"loss": loss}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=params['iter_prints'])

        estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(input=logits, axis=1))
        tf.summary.scalar('train_accuracy', accuracy[1])
    # Add evaluation metrics (for EVAL mode)
    else:
        eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
        estimatorSpec= tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return estimatorSpec


def main(model_dir='/tmp/mnist_networks/Q2'):
    params = {
        # Q1b
        'batch_size': 100,
        # Q1c
        'steps': 5000,
        # Q1d
        'learning_rate': 0.01,
        # Q1e
        'iter_prints': 250,
        # Q1f / Q5b
        'optimizer': 'sgb',

        # Q2
        'input_layer_dim': [-1, 28, 28, 1],
        'conv_dim': [5, 5],
        'pool_dim': [2, 2],
        'dense_units': 1024,

        # Q3
        # 'dropout_rate': 0.4,

        # Q4a
        # 'valid_prec': 0.2,

        # Q4b
        # 'early_stopping_rounds' : 3,

        # Q5a
        # 'lr_reduce_every_n' : 400,
        # 'lr_reduce_by' : 0.5

    }

    # Q1a - Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Q4a
    # valid_len = int((len(train_data) + len(eval_data)) * params['valid_prec'])
    # ind = np.arange(eval_data.shape[0])
    # np.random.shuffle(ind)
    # train_ind, valid_ind = ind[:len(ind) - valid_len], ind[len(ind) - valid_len:]
    # train_data, train_labels, valid_data, valid_labels = train_data[train_ind], train_labels[train_ind],\
    #                                                      train_data[valid_ind], train_labels[valid_ind]

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,  params=params,
                                              config=tf.contrib.learn.RunConfig(
                                                  save_checkpoints_steps=1,
                                                  save_summary_steps=250),
                                                    model_dir=model_dir
                                              )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=params['batch_size'],
        num_epochs=None,
        shuffle=True)

    # Q4a
    # valid_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": valid_data},
    #     y=valid_labels,
    #     batch_size=params['batch_size'],
    #     num_epochs=1,
    #     shuffle=False)

    # Q4b
    # validation_monitor = ValidationMonitor(
    #     every_n_steps=1,
    #     input_fn = valid_input_fn,
    #     early_stopping_metric="loss",
    #     early_stopping_metric_minimize=True,
    #     early_stopping_rounds=params['early_stopping_rounds']
    # )

    # list_of_monitors_and_hooks = [validation_monitor]
    # hooks = replace_monitors_with_hooks(list_of_monitors_and_hooks, mnist_classifier)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=params['steps'],
        # hooks=hooks
    )

    # Q1f Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
