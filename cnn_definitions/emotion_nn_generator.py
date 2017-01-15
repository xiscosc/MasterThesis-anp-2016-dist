from __future__ import absolute_import

import tensorflow as tf
from cnn_definitions.cnn_generator import fully_connected_layer


def emotion_classifier(logits_anp, weight_decay_rate, n_units):
    """
    Generates NN to perform emotion classification
    :param logits_anp: output of the CNN
    :param weight_decay_rate: weight decay rate
    :param n_units: list of integers with the number of units for each layer. len(n_units) layers will be created.
    :return:
    """
    if not isinstance(n_units, list):
        n_units = [n_units]

    net = logits_anp
    for i, n in enumerate(n_units[0:-1]):
        with tf.variable_scope('emotion_hidden'):
            net = fully_connected_layer(net, n, weight_decay_rate, 'fc%d' % (i+1))
            net = tf.nn.relu(net, name='relu%d' % (i+1))

    with tf.variable_scope('emotion_linear'):
        out = fully_connected_layer(net, n_units[-1], weight_decay_rate, 'classif')

    return out
