from __future__ import absolute_import

import sys
import pickle
import numpy as np
import tensorflow as tf

from helpers.monitoring import Timer


class CaffeModel:
    def __init__(self, weight_file_path, n_labels, image_mean=(103.939, 116.779, 123.68)):
        self.image_mean = image_mean
        self.n_labels = n_labels

        with open(weight_file_path, 'rb') as f, Timer('Loading pickle file'):
            if sys.version[0] == '3': self.pretrained_weights = pickle.load(f, encoding='latin1')
            else: self.pretrained_weights = pickle.load(f)

    def get_weight(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    def get_conv_weight(self, name):
        f = self.get_weight(name)
        return f.transpose((2, 3, 1, 0))

    def conv_layer(self, bottom, name, weight_decay_rate=None):
        with tf.variable_scope(name) as scope:
            w = self.get_conv_weight(name)
            b = self.get_bias(name)

            conv_weights = tf.get_variable(
                "W",
                shape=w.shape,
                initializer=tf.constant_initializer(w)
            )
            conv_biases = tf.get_variable(
                "b",
                shape=b.shape,
                initializer=tf.constant_initializer(b)
            )

            if weight_decay_rate is not None:
                wd_loss = tf.mul(tf.nn.l2_loss(conv_weights), weight_decay_rate, name='weight_loss')
                tf.add_to_collection('losses', wd_loss)

            conv = tf.nn.conv2d(bottom, conv_weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias, name=name)

        return relu

    def fc_layer(self, bottom, name, weight_decay_rate=None):
        shape = bottom.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(bottom, [-1, dim])

        cw = self.get_weight(name)
        b = self.get_bias(name)

        if name == "fc6":
            cw = cw.reshape((4096, 512, 7, 7))
            cw = cw.transpose((2, 3, 1, 0))
            cw = cw.reshape((25088, 4096))
        else:
            cw = cw.transpose((1, 0))

        with tf.variable_scope(name):
            cw = tf.get_variable(
                "W",
                shape=cw.shape,
                initializer=tf.constant_initializer(cw))
            b = tf.get_variable(
                "b",
                shape=b.shape,
                initializer=tf.constant_initializer(b))

            if weight_decay_rate is not None:
                wd_loss = tf.mul(tf.nn.l2_loss(cw), weight_decay_rate, name='weight_loss')
                tf.add_to_collection('losses', wd_loss)

            fc = tf.nn.bias_add(tf.matmul(x, cw), b, name=name)

        return fc

    @staticmethod
    def new_conv_layer(bottom, filter_shape, name, weight_decay_rate=None):
        with tf.variable_scope(name):
            w = tf.get_variable(
                "W",
                shape=filter_shape,
                initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))

            if weight_decay_rate is not None:
                wd_loss = tf.mul(tf.nn.l2_loss(w), weight_decay_rate, name='weight_loss')
                tf.add_to_collection('losses', wd_loss)

            conv = tf.nn.conv2d(bottom, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

        return bias  # relu

    @staticmethod
    def new_fc_layer(bottom, input_size, output_size, name, weight_decay_rate=None):
        shape = bottom.get_shape() 
        dim = np.prod(shape[1].value)
        x = tf.reshape(bottom, [-1, dim])

        with tf.variable_scope(name):
            w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))

            if weight_decay_rate is not None:
                wd_loss = tf.mul(tf.nn.l2_loss(w), weight_decay_rate, name='weight_loss')
                tf.add_to_collection('losses', wd_loss)

            fc = tf.nn.bias_add(tf.matmul(x, w), b, name=name)

        return fc
