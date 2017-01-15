from __future__ import absolute_import

from cnn_definitions.caffe_to_tf import CaffeModel

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as pretrained_nets


def resnet_v1_50_slim(inputs, num_classes, scope=None, reuse=None, is_training=False, weight_decay_rate=0.0001,
                      batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True, fcn=False):
    resnet_v1 = pretrained_nets.resnet_v1
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=is_training,
                                                   weight_decay=weight_decay_rate,
                                                   batch_norm_decay=batch_norm_decay,
                                                   batch_norm_epsilon=batch_norm_epsilon,
                                                   batch_norm_scale=batch_norm_scale)):
        logits, end_points = resnet_v1.resnet_v1_50(inputs,
                                                    num_classes=num_classes,
                                                    global_pool=True,
                                                    output_stride=None,
                                                    reuse=reuse,
                                                    scope=scope)
        if not fcn:
            logits = tf.reduce_mean(logits, [1, 2])
    return logits, end_points


def vgg16(inputs, num_classes, weights_path, scope=None, weight_decay_rate=None):
    """
    Creates VGG16 network. If scopes is a list, it returns a list of towers that share parameters.
    :param inputs: input placeholder
    :param num_classes: number of output neurons
    :param weights_path: path to the pickle file containing the Caffe weights
    :param scope: scope for the network
    :return: (logits, end_points) tuple
    """
    caffemodel = CaffeModel(weight_file_path=weights_path, n_labels=num_classes)

    def vgg16_inference():
        rgb = inputs * 255.
        r, g, b = tf.split(3, 3, rgb)
        bgr = tf.concat(3,
                        [
                            b - caffemodel.image_mean[0],
                            g - caffemodel.image_mean[1],
                            r - caffemodel.image_mean[2]
                        ])

        relu1_1 = caffemodel.conv_layer(bgr, "conv1_1", weight_decay_rate=weight_decay_rate)
        relu1_2 = caffemodel.conv_layer(relu1_1, "conv1_2", weight_decay_rate=weight_decay_rate)

        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

        relu2_1 = caffemodel.conv_layer(pool1, "conv2_1", weight_decay_rate=weight_decay_rate)
        relu2_2 = caffemodel.conv_layer(relu2_1, "conv2_2", weight_decay_rate=weight_decay_rate)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = caffemodel.conv_layer(pool2, "conv3_1", weight_decay_rate=weight_decay_rate)
        relu3_2 = caffemodel.conv_layer(relu3_1, "conv3_2", weight_decay_rate=weight_decay_rate)
        relu3_3 = caffemodel.conv_layer(relu3_2, "conv3_3", weight_decay_rate=weight_decay_rate)
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        relu4_1 = caffemodel.conv_layer(pool3, "conv4_1", weight_decay_rate=weight_decay_rate)
        relu4_2 = caffemodel.conv_layer(relu4_1, "conv4_2", weight_decay_rate=weight_decay_rate)
        relu4_3 = caffemodel.conv_layer(relu4_2, "conv4_3", weight_decay_rate=weight_decay_rate)
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

        relu5_1 = caffemodel.conv_layer(pool4, "conv5_1", weight_decay_rate=weight_decay_rate)
        relu5_2 = caffemodel.conv_layer(relu5_1, "conv5_2", weight_decay_rate=weight_decay_rate)
        relu5_3 = caffemodel.conv_layer(relu5_2, "conv5_3", weight_decay_rate=weight_decay_rate)
        pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')

        fc6 = caffemodel.fc_layer(pool5, "fc6", weight_decay_rate=weight_decay_rate)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc7 = caffemodel.fc_layer(relu6, "fc7", weight_decay_rate=weight_decay_rate)
        relu7 = tf.nn.relu(fc7, name="relu7")
        logits = caffemodel.new_fc_layer(relu7, 4096, num_classes, "fc8")

        end_points = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'pool1': pool1,
                      'relu2_1': relu2_1, 'relu2_2': relu2_2, 'pool2': pool2,
                      'relu3_1': relu3_1, 'relu3_2': relu3_2, 'pool3': pool3,
                      'relu4_1': relu4_1, 'relu4_2': relu4_2, 'pool4': pool4,
                      'relu5_1': relu5_1, 'relu5_2': relu5_2, 'pool5': pool5,
                      'fc6': fc6, 'relu6': relu6, 'fc7': fc7, 'relu7': relu7,
                      'logits': logits}

        return logits, end_points

    '''
    if isinstance(scope, list):
        towers = list()
        for sc in scope:
            with tf.variable_scope(sc):
                towers.append(vgg16_inference())
        return towers
    else:
        with tf.variable_scope(scope):
            return vgg16_inference()
    '''

    return vgg16_inference()


def resnet_v1_50(inputs, num_classes, collection_name='resnet_weights', is_training=False, preprocess=False):
    """
    Creates graph for ResNet50. Returns the logits and Global Average Pooling (GAP) layers.
    If the pre-trained model is used, the expected input is BGR [0, 255] with mean subtracted.
    :param inputs:
    :param num_classes:
    :param collection_name: name of the collection where all the weights will be stored
    :param is_training:
    :param preprocess: pre-process images read from TFRecords (RGB [0,1]) to fulfill the input requirements
    of the pre-trained model
    :return: logits, gap
    """
    from cnn_definitions import resnet
    logits, gap = resnet.inference(inputs,
                                   num_classes=num_classes,
                                   is_training=is_training,
                                   collection_name=collection_name,
                                   preprocess=preprocess)
    return logits, gap


def _get_variables_to_restore(scope, restore_logits):
    variables_to_restore = []
    for op in slim.get_model_variables(scope):
        if restore_logits or not op.name.__contains__('logits'): variables_to_restore.append(op)
    return variables_to_restore


def initialize_from_checkpoint(sess, checkpoint_path, scope, restore_logits, model_generation_func, verbose=False):
    """
    Restores the model variables.
    :param sess: current session.
    :param checkpoint_path: path to the checkpoint file.
    :param restore_logits: boolean. Determines if the logits are also restored.
    :param scope: scope of the variables to be restored.
    :param model_generation_func: function used to generate the graph for the pre-trained model.
    :param verbose: boolean. Prints information of the process.
    :return:
    """
    if model_generation_func == resnet_v1_50:
        variables_to_restore = _get_variables_to_restore(scope, restore_logits)
        if verbose:
            print('Variables to restore: ')
            for op in variables_to_restore: print('\t%s' % op.name)
        saver = tf.train.Saver(variables_to_restore)
        for op in variables_to_restore: tf.logging.info('\tRESTORED: %s' % op.name)
        return saver

        # saver.restore(sess, checkpoint_path)
        # print('\nRestored variables:')

    else:
        raise ValueError('The specified model_generation_func is not supported')


def fully_connected_layer(input_tensor, num_output_units, weight_decay, name):
    num_input_units = input_tensor.get_shape()[1]

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    weights = tf.get_variable('%s/weights' % name,
                              shape=[num_input_units, num_output_units],
                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                              dtype=tf.float32,
                              regularizer=regularizer,
                              trainable=True)
    biases = tf.get_variable('%s/bias' % name,
                             shape=[num_output_units],
                             initializer=tf.zeros_initializer,
                             dtype=tf.float32,
                             trainable=True)
    fc = tf.nn.xw_plus_b(input_tensor, weights, biases)
    return fc


def projection_matrix(input_tensor, num_output_units, weight_decay, name):
    num_input_units = input_tensor.get_shape()[1].value

    input_tensor = tf.reshape(input_tensor, [-1, num_input_units])

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    weights = tf.get_variable('%s/weights' % name,
                              shape=[num_input_units, num_output_units],
                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                              dtype=tf.float32,
                              regularizer=regularizer,
                              trainable=True)
    fc = tf.matmul(input_tensor, weights)
    return fc
