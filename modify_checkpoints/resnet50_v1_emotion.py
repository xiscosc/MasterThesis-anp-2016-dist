"""
Loads the pre-trained ResNet50 model from TF-Slim and stores it with a new scope
"""


from __future__ import absolute_import

import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(0, '..')
from cnn_definitions import cnn_generator
from modify_checkpoints.modify_scope import assign_to_new_scope

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')

logits, _ = cnn_generator.resnet_v1_50_slim(inputs, 1000,
                                            scope='resnet_v1_50',
                                            reuse=None,
                                            is_training=True,
                                            weight_decay_rate=0.0005,
                                            batch_norm_decay=0.997,
                                            batch_norm_epsilon=1e-5,
                                            batch_norm_scale=True)

logits_emotion, _ = cnn_generator.resnet_v1_50_slim(inputs, 8,
                                                    scope='emotion_resnet50',
                                                    reuse=None,
                                                    is_training=True,
                                                    weight_decay_rate=0.0005,
                                                    batch_norm_decay=0.997,
                                                    batch_norm_epsilon=1e-5,
                                                    batch_norm_scale=True)

sess = tf.InteractiveSession()


# Restore pre-trained model with the original scope
variables_to_restore = []
for op in slim.get_model_variables('resnet_v1_50'):
    if not op.name.__contains__('logits'): variables_to_restore.append(op)

saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50.ckpt')

# Assign and save weights
sess.run(assign_to_new_scope('emotion_resnet50', 'resnet_v1_50', variables_to_restore))

emotion_variables = []
for op in slim.get_model_variables('emotion_resnet50'):
    if not op.name.__contains__('logits'): emotion_variables.append(op)


saver_anp = tf.train.Saver(emotion_variables)
saver_anp.save(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50_emotion.ckpt')
