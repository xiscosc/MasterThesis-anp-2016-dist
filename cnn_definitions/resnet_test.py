from __future__ import absolute_import

import argparse
import numpy as np
import tensorflow as tf

from cnn_definitions import cnn_generator
from helpers.monitoring import Timer
from ground_truth.imagenet_synset import synset

parser = argparse.ArgumentParser(description='Generates random stratified train/val splits for MVSO-EN')
parser.add_argument('--image_index', dest='image_index', type=int, default=0,
                    help='Fraction of the dataset used for training')
args = parser.parse_args()


# Variables
filename = '/home/vcampos/NetworkDrives/bsc/mvso_en_1200/val/val-00100-of-01000'
weights_collection_name = 'anp_resnet'
is_training = False


with Timer('Loading image from %s' % filename):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([3], dtype=tf.int64,
                                                default_value=[-1] * 3),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                           default_value=-1),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                          default_value=-1),
    }

    it = tf.python_io.tf_record_iterator(filename)

    for i in xrange(args.image_index):
        it.next()

    features = tf.parse_single_example(it.next(), feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    image = tf.reshape(image, [height[0], width[0], 3])

    image_batch = tf.expand_dims(image, 0)


inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
# net, end_points = cnn_generator.resnet_v1_50(inputs, 1000, scope='noun', is_training=True, reuse=None)

with Timer('Creating ResNet50 graph'):
    logits, gap = cnn_generator.resnet_v1_50(inputs, 1000, collection_name=weights_collection_name, is_training=is_training,
                                             preprocess=True)

print '\nResNet50 variables: '
for op in tf.get_collection(weights_collection_name):
    print '\t%s' % op.name

if is_training:
    variables_to_restore = [op for op in tf.get_collection(weights_collection_name) if not op.name.__contains__('fc')]
else:
    variables_to_restore = [op for op in tf.get_collection(weights_collection_name)]

sess = tf.InteractiveSession()
summary_writer = tf.train.SummaryWriter('/tmp/tensorboard_logs/')
im_sum_op = tf.image_summary('images', image_batch)


with Timer('Restoring variables'):
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, '/home/vcampos/Downloads/tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt')


im_sum_val = sess.run(im_sum_op)
summary_writer.add_summary(im_sum_val)

image_batch = tf.image.resize_images(image_batch, 224, 224)

with Timer('Inference'):
    output = sess.run(logits, feed_dict={inputs: image_batch.eval()})

print synset[np.argmax(output, axis=1)[0]]

sess.close()
