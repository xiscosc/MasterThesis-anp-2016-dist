from __future__ import absolute_import

import os
import re
import sys
import time
from datetime import datetime
import subprocess
import numpy as np
import tensorflow as tf

from train import batch_generator_mvso
from train.mvso_data import MVSOData
from cnn_definitions import cnn_generator, vgg_preprocessing
from train.embedding_utils import *

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/bsc31/bsc31953/tensorflow_training/mvso/',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1800,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01, """Initial learning rate value.""")
tf.app.flags.DEFINE_boolean('exponential_decay', False, """Whether to use exponential LR decay.""")
tf.app.flags.DEFINE_integer('lr_epochs_per_decay', 10, """Number of epochs between lr decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.9, """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, """Moving average decay.""")
tf.app.flags.DEFINE_float('weight_decay_rate', 0.0001, """Weight decay rate.""")
tf.app.flags.DEFINE_string('cnn', 'resnet50', """CNN architecture""")
tf.app.flags.DEFINE_boolean('remove_dir', False,
                            """Whether to remove train_dir before starting training.""")
tf.app.flags.DEFINE_string('checkpoint', None, """Checkpoint file with the pre-trained model weights""")
tf.app.flags.DEFINE_boolean('restore_logits', False, """Whether to restore logits when loading a pre-trained model.""")
tf.app.flags.DEFINE_string('optimizer', 'SGD', """SGD, Momentum, RMSProp, Adam""")
tf.app.flags.DEFINE_boolean('resume_training', False, """Resume training from last checkpoint in train_dir""")
tf.app.flags.DEFINE_boolean('histograms', False, """Whether to store variable histograms summaries.""")
tf.app.flags.DEFINE_integer('eval_interval_iters', 3000, """How often to run the eval, in training steps.""")

# Job file to submit after finishing
tf.app.flags.DEFINE_string('evaluation_job', None, """Path to the cmd file that performs the evaluation""")
tf.app.flags.DEFINE_string('resume_job', None, """Path to the cmd file to be submitted after finishing""")

# Loss function
tf.app.flags.DEFINE_string('anp_vectors', None, """Pickle file with all the word2vec vectors for ANPs""")
tf.app.flags.DEFINE_boolean('label_smoothing', False, """Whether to use similarities instead of one-hot encoding.""")

#Distributed flags
tf.app.flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of hostname:port')
tf.app.flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of hostname:port')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('worker_url', None, 'Worker GRPC URL')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_boolean('sync_replicas', False, """Use SyncReplicasOptimizer""")
tf.app.flags.DEFINE_integer('backup_workers', 0, 'Number of backup workers')
is_chief = (FLAGS.task_index == 0)
slim = tf.contrib.slim
num_workers = len(FLAGS.worker_hosts.split(','))

def tower_loss(scope, reuse):
    """
    Calculate the total loss on a single tower.
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels
    dataset = MVSOData(subset='train')
    images, labels = batch_generator_mvso.generate_batch(dataset,
                                                         FLAGS.batch_size,
                                                         train=True,
                                                         image_processing_fn=vgg_preprocessing.preprocess_image,
                                                         num_preprocess_threads=8)

    if FLAGS.histograms:
        tf.summary.histogram('preprocessed_input', images)
        tf.summary.histogram('labels', labels)

    if FLAGS.label_smoothing:
        assert FLAGS.anp_vectors is not None, 'Label smoothing requires from the ANP vectors'
        smoothed_labels_matrix = generate_label_smoothing(FLAGS.anp_vectors, 'smoothed_labels_matrix')
    else:
        smoothed_labels_matrix = None
    # Build inference Graph.
    if FLAGS.cnn == 'resnet50':
        logits, _ = cnn_generator.resnet_v1_50_slim(images, dataset.num_classes()[0],
                                                    scope='anp_resnet50',
                                                    reuse=reuse,
                                                    is_training=True,
                                                    weight_decay_rate=FLAGS.weight_decay_rate,
                                                    batch_norm_decay=0.997,
                                                    batch_norm_epsilon=1e-5,
                                                    batch_norm_scale=True)
    else:
        raise ValueError('The specified CNN architecture is not supported')

    tf.summary.histogram('predicted_class', tf.arg_max(logits, 1))

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logits, labels, smoothed_labels_matrix)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        # loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
        loss_name = l.op.name
        tf.logging.info('Creating summary for %s', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
       # tf.summary.scalar(loss_name +' (raw)', l)
       # tf.summary.scalar(loss_name, loss_averages.average(l))

    #with tf.control_dependencies([loss_averages_op]):
    #    total_loss = tf.identity(total_loss)
    return total_loss, logits, labels


def loss(logits, labels, smoothed_labels_matrix=None):
    """
    Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
      smoothed_labels_matrix: matrix with the normalized similarity vectors for each class
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    if not FLAGS.label_smoothing:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    else:
        labels = tf.gather(smoothed_labels_matrix, labels, name='label_smoothing')
        normalized_logits = tf.nn.softmax(logits, name='softmax')
        cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.log(tf.clip_by_value(normalized_logits, 1e-9, 1.0))), 1,
                                       name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(cluster, server):
    device_setter = tf.train.replica_device_setter(cluster=cluster,
                                                   worker_device="/job:worker/task:%d" % FLAGS.task_index)

    with tf.device(device_setter):
        """Train for a number of steps."""
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        # global_step = tf.get_variable(
        #     'global_step', [],
        #    initializer=tf.constant_initializer(0), trainable=False)

        with tf.device('/job:ps/cpu:0'):
            collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
            global_step = tf.get_variable(tf.GraphKeys.GLOBAL_STEP,
                                          shape=[], dtype=tf.int64,
                                          initializer=tf.zeros_initializer(),
                                          regularizer=None,
                                          trainable=False,
                                          collections=collections)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (MVSOData(subset='train').num_examples_per_epoch() / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.lr_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        if FLAGS.exponential_decay:
            tf.logging.info('Using exponential learning rate decay')
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.lr_decay_factor,
                                            staircase=True)
        else:
            tf.logging.info('Using constant learning rate of %f', FLAGS.initial_learning_rate)
            lr = tf.get_variable(
                'learning_rate', [],
                initializer=tf.constant_initializer(FLAGS.initial_learning_rate), trainable=False)

        # Create an optimizer that performs gradient descent.
        if FLAGS.optimizer.lower() == 'sgd':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif FLAGS.optimizer.lower() == 'momentum':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif FLAGS.optimizer.lower() == 'rmsprop':
            # Parameters from 'Rethinking Inception Architecture for Computer Vision' (except for the LR)
            opt = tf.train.RMSPropOptimizer(lr, momentum=0.9, epsilon=1.0)
        elif FLAGS.optimizer.lower() == 'adam':
            opt = tf.train.AdamOptimizer(lr)
        else:
            raise AttributeError('The specified optimizer is not supported')


        if FLAGS.sync_replicas:
            replicas = num_workers - FLAGS.backup_workers
            if replicas < 1:
                raise ValueError('Too much backup workers')
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=replicas,
                                       total_num_replicas=num_workers)


        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_logits = []
        tower_labels = []
        tower_losses = []
        reuse = None
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in range(FLAGS.num_gpus):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    with tf.device('/job:worker/GPU:%d' % i):
                        # Calculate the loss for one tower. This function constructs
                        # the entire model but shares the variables across all towers.
                        loss, logits, labels = tower_loss(scope, reuse)

                        # Reuse variables for the next tower.
                        reuse = True
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(loss)
                        capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in grads]
                        # Keep track of the gradients across all towers.
                        tower_grads.append(capped_gvs)
                        tower_logits.append(logits)
                        tower_labels.append(labels)
                        tower_losses.append(loss)

        # Concatenate the outputs of all towers
        logits_op = tf.concat(axis=0, values=tower_logits, name='concat_logits')
        labels_op = tf.concat(axis=0, values=tower_labels, name='concat_labels')
        loss_op = tf.reduce_mean(tower_losses)

        # Update BN's moving_mean and moving_variance
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            tf.logging.info('Gathering update_ops')
            with tf.control_dependencies(tf.tuple(update_ops)):
                loss_op = tf.identity(loss_op)

        # Track the loss of all towers
        summaries.append(tf.summary.scalar('combined_loss', loss_op))

        # Compute top-1 accuracy
        top1_predictions_op = tf.argmax(logits_op, 1)
        top1_correct_pred = tf.equal(top1_predictions_op, tf.cast(labels_op, tf.int64))
        top1_accuracy_op = tf.reduce_mean(tf.cast(top1_correct_pred, tf.float32))

        # Compute top-5 accuracy
        _, top5_predictions_op = tf.nn.top_k(logits_op, k=5)
        top5_correct_pred = tf.cast(tf.equal(tf.transpose(top5_predictions_op), labels_op), tf.float32)
        top5_accuracy_op = tf.reduce_mean(tf.reduce_sum(top5_correct_pred, 0), 0)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        if FLAGS.histograms:
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for op in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.logging.info(op.name)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        if FLAGS.histograms:
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver_cp = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.

        if not FLAGS.resume_training and  FLAGS.checkpoint is not None:
            tf.logging.info('INIT RESTORE FROM CHECKPOINT')
            saver = cnn_generator.initialize_from_checkpoint(sess=None,
                                                     checkpoint_path=FLAGS.checkpoint,
                                                     scope='anp_resnet50',
                                                     restore_logits=FLAGS.restore_logits,
                                                     model_generation_func=cnn_generator.resnet_v1_50)

        scaffold = tf.train.Scaffold(
            saver=saver_cp,
            init_op=init,
        )
        hooks = []
        if FLAGS.sync_replicas:
            hooks.append(opt.make_session_run_hook(is_chief))

        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.train.MonitoredTrainingSession(is_chief=is_chief,
                                               checkpoint_dir=FLAGS.train_dir,
                                               save_summaries_steps=100,
                                               scaffold=scaffold,
                                               master=server.target,
                                               hooks=hooks,
                                               config=config) as sess:

            tf.logging.info('%s %s %d: Session initialization complete.' %
                            (datetime.now(), FLAGS.job_name, FLAGS.task_index))


            if not FLAGS.resume_training and FLAGS.checkpoint is not None:
                saver.restore(sess, FLAGS.checkpoint)

            if is_chief:
                summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            accumulated_top1_accuracy_10_steps = 0.
            accumulated_top1_accuracy_100_steps = 0.
            accumulated_top5_accuracy_10_steps = 0.
            accumulated_top5_accuracy_100_steps = 0.

            for step in range(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value, top1_accuracy_value, top5_accuracy_value = sess.run([train_op, loss_op,
                                                                                    top1_accuracy_op,
                                                                                    top5_accuracy_op])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                accumulated_top1_accuracy_10_steps += top1_accuracy_value
                accumulated_top1_accuracy_100_steps += top1_accuracy_value
                accumulated_top5_accuracy_10_steps += top5_accuracy_value
                accumulated_top5_accuracy_100_steps += top5_accuracy_value

                if step == 0:
                    continue

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus

                    format_str = 'WORKER %d | %s: step %d, global_step = %d, loss = %.2f, top-1 = %.3f%%, top-5 = %.3f%% ' \
                                 '(%.1f examples/sec; %.3f sec/batch)'
                    tf.logging.info(format_str % (FLAGS.task_index, datetime.now(), step, int(global_step.eval(session=sess)), loss_value,
                                                  accumulated_top1_accuracy_10_steps * 10,
                                                  accumulated_top5_accuracy_10_steps * 10,
                                                  examples_per_sec, sec_per_batch))
                    accumulated_top1_accuracy_10_steps = 0.
                    accumulated_top5_accuracy_10_steps = 0.

                if step % 100 == 0 and is_chief:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step.eval(session=sess) - 1)
                    # examples_per_sec_summary = sess.run(tf.scalar_summary('Examples per second', examples_per_sec))
                    # summary_writer.add_summary(examples_per_sec_summary, global_step.eval() - 1)
                    # top1_acc_summary = sess.run(tf.scalar_summary('accuracy_top-1 (%)',
                    #                                               accumulated_top1_accuracy_100_steps))
                    # summary_writer.add_summary(top1_acc_summary, global_step.eval() - 1)
                    # top5_acc_summary = sess.run(tf.scalar_summary('accuracy_top-5 (%)',
                    #                                               accumulated_top5_accuracy_100_steps))
                    # summary_writer.add_summary(top5_acc_summary, global_step.eval() - 1)
                    accumulated_top1_accuracy_100_steps = 0.
                    accumulated_top5_accuracy_100_steps = 0.

                if is_chief and (step % FLAGS.eval_interval_iters == 0 or (step + 1) == FLAGS.max_steps):
                    if FLAGS.evaluation_job:
                        jobname = FLAGS.evaluation_job
                    else:
                        num = num_workers
                        jobname = ("/gpfs/projects/bsc31/bsc31953/DISTRIBUTED/eval%d.cmd" % num)
                    message = "STEP %d EVAL JOB %s sent" % (step, jobname)
                    tf.logging.info(message)
                    subprocess.check_output(['mnsubmit', jobname])


def main(argv=None):

    #Cluster declaration
    cluster = tf.train.ClusterSpec({'ps': FLAGS.ps_hosts.split(','),
                                    'worker': FLAGS.worker_hosts.split(',')})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)


    if FLAGS.job_name == 'ps':
        tf.logging.info('%s %s %d: JOINING TO SERVER.' %
                        (datetime.now(), FLAGS.job_name, FLAGS.task_index))
        server.join()
    elif FLAGS.job_name == 'worker':
        tf.logging.info('%s %s %d: STARTING TRAINING.' %
                        (datetime.now(), FLAGS.job_name, FLAGS.task_index))
        if is_chief:
            if tf.gfile.Exists(FLAGS.train_dir) and FLAGS.remove_dir:
                tf.gfile.DeleteRecursively(FLAGS.train_dir)
            if not tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.MakeDirs(FLAGS.train_dir)
        train(cluster, server)

        if FLAGS.resume_job is not None:
            job_files_path = FLAGS.resume_job.rsplit('/', 1)[0]
            with open(os.path.join(job_files_path, 'resume_params.txt'), 'r') as f:
                lines = f.readlines()
            if lines[0].strip() == 'resume':
                os.environ['LEARNING_RATE'] = str(lines[1].strip())
                subprocess.check_output(['mnsubmit', FLAGS.resume_job], env=dict(os.environ))


if __name__ == '__main__':
    tf.app.run()
