# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Evaluates models generated during training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import pickle

import numpy as np
import tensorflow as tf

from train import batch_generator_mvso
from train.mvso_data import MVSOData
from cnn_definitions import cnn_generator, vgg_preprocessing


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', None,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', None,
                           """Either 'train' or 'val'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', None,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Set to 0 to evaluate the whole set.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, """Moving average decay.""")
tf.app.flags.DEFINE_string('cnn', 'resnet50', """CNN architecture""")
tf.app.flags.DEFINE_boolean('remove_dir', False, """Whether to remove eval_dir before starting evaluation.""")

# Store outputs
tf.app.flags.DEFINE_string('logits_output_file', None,
                           """File where a pickled list with (logits, ground_truth) tuples for each image will be stored""")


def eval_once(saver, summary_writer, top_1_op, top_5_op, top_10_op, top_100_op, summary_op, num_examples,
              logits_op, labels_op, filenames_op):
    """
    Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      top_10_op: Top 10 op.
      top_100_op: Top 100 op.
      summary_op: Summary op.
      num_examples: number of samples in the evaluation set
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Restoring from %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            tf.logging.info('No checkpoint file found')
            return

        # Store the outputs if requested
        if FLAGS.logits_output_file is not None:
            results_dict = {}

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            true_count_top5 = 0  # Counts the number of top-5 correct predictions.
            true_count_top10 = 0  # Counts the number of top-10 correct predictions.
            true_count_top100 = 0  # Counts the number of top-100 correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            while step < num_iter and not coord.should_stop():
                if FLAGS.logits_output_file is None:
                    predictions, predictions_top5, predictions_top10, predictions_top100 = \
                        sess.run([top_1_op, top_5_op, top_10_op, top_100_op])
                else:
                    predictions, predictions_top5, predictions_top10, predictions_top100, logits, labels, filenames =\
                        sess.run([top_1_op, top_5_op, top_10_op, top_100_op, logits_op, labels_op, filenames_op])
                    for i in range(logits.shape[0]):
                        results_dict[filenames[i]] = (logits[i, :], labels[i])
                true_count += np.sum(predictions)
                true_count_top5 += np.sum(predictions_top5)
                true_count_top10 += np.sum(predictions_top10)
                true_count_top100 += np.sum(predictions_top100)
                step += 1
                tf.logging.info(('Step: %d/%d  --  top-1 accuracy: %.3f%%\ttop-5 accuracy:%.3f%%'
                                 '\ttop-10 accuracy: %.3f%%\ttop-100 accuracy:%.3f%%')
                                % (step, num_iter,
                                   100. * true_count / (step * FLAGS.batch_size),
                                   100. * true_count_top5 / (step * FLAGS.batch_size),
                                   100. * true_count_top10 / (step * FLAGS.batch_size),
                                   100. * true_count_top100 / (step * FLAGS.batch_size)))

            # Compute precision @ 1, 5, 10, 100.
            precision = true_count / total_sample_count
            precision_top5 = true_count_top5 / total_sample_count
            precision_top10 = true_count_top10 / total_sample_count
            precision_top100 = true_count_top100 / total_sample_count
            tf.logging.info('%s: top-1 accuracy: %.3f\ttop-5 accuracy:%.3f\ttop-10 accuracy:%.3f\ttop-100 accuracy:%.3f'
                            % (datetime.now(), precision, precision_top5, precision_top10, precision_top100))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='accuracy_top-1 (%)', simple_value=100. * precision)
            summary.value.add(tag='accuracy_top-5 (%)', simple_value=100. * precision_top5)
            summary.value.add(tag='accuracy_top-10 (%)', simple_value=100. * precision_top10)
            summary.value.add(tag='accuracy_top-100 (%)', simple_value=100. * precision_top100)
            summary_writer.add_summary(summary, global_step)

            if FLAGS.logits_output_file is not None:
                with open(FLAGS.logits_output_file, 'wb') as f:
                    pickle.dump(results_dict, f, protocol=0)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Evaluate MVSO for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels
        dataset = MVSOData(subset=FLAGS.eval_data)
        if FLAGS.logits_output_file is not None:
            images, labels, filenames = batch_generator_mvso.generate_batch(dataset, FLAGS.batch_size, train=False,
                                                                       image_processing_fn=vgg_preprocessing.preprocess_image,
                                                                       include_filename=True)
        else:
            images, labels = batch_generator_mvso.generate_batch(dataset, FLAGS.batch_size, train=False,
                                                                 image_processing_fn=vgg_preprocessing.preprocess_image)
            filenames = None

        # Build a Graph that computes the logits predictions from the inference model.
        if FLAGS.cnn == 'resnet50':
            logits, _ = cnn_generator.resnet_v1_50_slim(images, dataset.num_classes()[0],
                                                        scope='anp_resnet50',
                                                        reuse=None,
                                                        is_training=False,
                                                        # weight_decay_rate=FLAGS.weight_decay_rate,
                                                        batch_norm_decay=0.997,
                                                        batch_norm_epsilon=1e-5,
                                                        batch_norm_scale=True)
        else:
            raise ValueError('The specified CNN architecture is not supported')

        # Calculate predictions.
        top_1_op = top_k_accuracy(logits, labels, 1)
        top_5_op = top_k_accuracy(logits, labels, 5)
        top_10_op = top_k_accuracy(logits, labels, 10)
        top_100_op = top_k_accuracy(logits, labels, 100)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        # tf.logging.info('Variables to restore:')
        # for op in variables_to_restore:
        #     tf.logging.info('\t%s' % op)

        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            if FLAGS.num_examples == 0:
                num_examples = dataset.num_examples_per_epoch()
            else:
                num_examples = FLAGS.num_examples
            eval_once(saver, summary_writer, top_1_op, top_5_op, top_10_op, top_100_op, summary_op, num_examples,
                      logits, labels, filenames)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def top_k_accuracy(logits, labels, k):
    _, topk_predictions_op = tf.nn.top_k(logits, k=k)
    topk_correct_pred = tf.cast(tf.equal(tf.transpose(topk_predictions_op), labels), tf.float32)
    return tf.reduce_sum(topk_correct_pred, 0)


def predicted_class(logits):
    return tf.argmax(logits, 1)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir) and FLAGS.remove_dir:
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
