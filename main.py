import os
import re
import sys
import time
from datetime import datetime
import subprocess
import numpy as np
import tensorflow as tf

from train import batch_generator_mvso_dist as batch_generator_mvso
from train.mvso_data import MVSOData
from cnn_definitions import cnn_generator, vgg_preprocessing
from train.embedding_utils import *
from deployment import model_deploy

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1800,
                            """Time between chekpoint saves.""")
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
is_chief = (FLAGS.task_index == 0)
slim = tf.contrib.slim


def _get_learning_rate(global_step):
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (MVSOData(subset='train').num_examples_per_epoch() / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.lr_epochs_per_decay)

    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

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

    return lr


def _get_optimizer(lr):
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

    return opt


def _train(cluster, server):
    device_setter = tf.train.replica_device_setter(cluster=cluster,
                                                   worker_device="/job:worker/task:%d" % FLAGS.task_index)

    with tf.device(device_setter):
        global_step = slim.get_or_create_global_step()

        lr = _get_learning_rate(global_step)
        opt = _get_optimizer(lr)
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)


        deployment = model_deploy.deploy()


def main(argv=None):
    # Cluster declaration
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
        _train(cluster, server)

        if FLAGS.resume_job is not None:
            job_files_path = FLAGS.resume_job.rsplit('/', 1)[0]
            with open(os.path.join(job_files_path, 'resume_params.txt'), 'r') as f:
                lines = f.readlines()
            if lines[0].strip() == 'resume':
                os.environ['LEARNING_RATE'] = str(lines[1].strip())
                subprocess.check_output(['mnsubmit', FLAGS.resume_job], env=dict(os.environ))


if __name__ == '__main__':
    tf.app.run()
