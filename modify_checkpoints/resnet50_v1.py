"""
Loads the pre-trained ResNet50 model from TF-Slim and stores it with a new scope
so it can be loaded in the 2-tower experiment
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

logits_anp, _ = cnn_generator.resnet_v1_50_slim(inputs, 1200,
                                                scope='anp_resnet50',
                                                reuse=None,
                                                is_training=True,
                                                weight_decay_rate=0.0005,
                                                batch_norm_decay=0.997,
                                                batch_norm_epsilon=1e-5,
                                                batch_norm_scale=True)

logits_noun, _ = cnn_generator.resnet_v1_50_slim(inputs, 617,
                                                 scope='noun_resnet50',
                                                 reuse=None,
                                                 is_training=True,
                                                 weight_decay_rate=0.0005,
                                                 batch_norm_decay=0.997,
                                                 batch_norm_epsilon=1e-5,
                                                 batch_norm_scale=True)

logits_adj, _ = cnn_generator.resnet_v1_50_slim(inputs, 350,
                                                scope='adj_resnet50',
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

# Assign and save weights for ANPNet, NounNet and AdjNet
sess.run(assign_to_new_scope('anp_resnet50', 'resnet_v1_50', variables_to_restore))
sess.run(assign_to_new_scope('noun_resnet50', 'resnet_v1_50', variables_to_restore))
sess.run(assign_to_new_scope('adj_resnet50', 'resnet_v1_50', variables_to_restore))

# assign_to_new_scope('anp_resnet50', 'resnet_v1_50', variables_to_restore)
# assign_to_new_scope('noun_resnet50', 'resnet_v1_50', variables_to_restore)
# assign_to_new_scope('adj_resnet50', 'resnet_v1_50', variables_to_restore)

anp_variables = []
for op in slim.get_model_variables('anp_resnet50'):
    if not op.name.__contains__('logits'): anp_variables.append(op)

noun_variables = []
for op in slim.get_model_variables('noun_resnet50'):
    if not op.name.__contains__('logits'): noun_variables.append(op)

adj_variables = []
for op in slim.get_model_variables('adj_resnet50'):
    if not op.name.__contains__('logits'): adj_variables.append(op)


saver_anp = tf.train.Saver(anp_variables)
saver_anp.save(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50_anp.ckpt')

saver_noun = tf.train.Saver(noun_variables)
saver_noun.save(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50_noun.ckpt')

saver_adj = tf.train.Saver(adj_variables)
saver_adj.save(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50_adj.ckpt')

saver_noun_adj = tf.train.Saver(noun_variables+adj_variables)
saver_noun_adj.save(sess, '/home/bsc31/bsc31953/cnn_models/resnet_v1_50_noun_adj.ckpt')
