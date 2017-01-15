import pickle
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def load_vectors(vectors_file, ground_truth_file, name, l2_normalize=True):
    """
    Loads word2vec vectors, normalizes them and creates a matrix where mat[i,:] is the i-th normalized vector
    :param vectors_file: path to the pickle file with the vectors
    :param ground_truth_file: path to the list of classes
    :param name: name of the variable containing the vectors
    :param l2_normalize: whether to l2-normalize the vectors
    :return: TF Constant with the vectors
    """
    # Load data from files
    with open(vectors_file, 'rb') as f:
        vectors_dict = pickle.load(f, encoding='latin1')
    with open(ground_truth_file, 'r') as f:
        ground_truth = [line.strip() for line in f.readlines()]

    # Get shape of the matrix
    n_classes = len(ground_truth)
    vector_dim = np.shape(vectors_dict[ground_truth[0]])[0]

    # Create list of vectors in the same order as the labels and normalize them
    if l2_normalize:
        all_vectors = [vectors_dict[label]/np.linalg.norm(vectors_dict[label]) for label in ground_truth]
    else:
        all_vectors = [vectors_dict[label] for label in ground_truth]

    # Create matrix with all the vectors
    vector_matrix = np.array(all_vectors)

    return tf.get_variable(name, [n_classes, vector_dim],
                           initializer=tf.constant_initializer(vector_matrix),
                           trainable=False)


def load_anp_vectors(vectors_file, name, l2_normalize=True):
    """
    Loads word2vec vectors, normalizes them and creates a matrix where mat[i,:] is the i-th normalized vector
    :param vectors_file: path to the pickle file with the vectors
    :param name: name of the variable containing the vectors
    :return: TF Constant with the vectors
    """
    # Load vectors
    with open(vectors_file, 'rb') as f:
        vector_matrix = pickle.load(f, encoding='latin1')

    # Normalize vectors
    if l2_normalize:
        for class_index in range(np.shape(vector_matrix)[0]):
            vector_matrix[class_index, :] /= np.linalg.norm(vector_matrix[class_index, :])

    return tf.get_variable(name, [vector_matrix.shape[0], vector_matrix.shape[1]],
                           initializer=tf.constant_initializer(vector_matrix),
                           trainable=False)


def get_predicted_class(outputs, vector_matrix, k=1, metric='cos'):
    if metric == 'cos':
        # [batch_size, vector_dim] * [vector_dim, num_classes] = [batch_size, num_classes]
        similarity_matrix = tf.matmul(outputs, vector_matrix, transpose_b=True)
    elif metric == 'l2':
        #similarity_matrix = 0. - tf.sqrt(tf.reduce_sum(tf.square(tf.sub(tf.expand_dims(vector_matrix, 0),
        #                                                                tf.expand_dims(outputs, 1))),
        #                                               reduction_indices=2))
        similarity_matrix = tf.matmul(tf.nn.l2_normalize(outputs, dim=1), tf.nn.l2_normalize(vector_matrix, dim=1),
                                      transpose_b=True)
    else:
        raise ValueError('Not supported distance metric: ', metric)

    # Find the class with the highest similarity
    values, indices = tf.nn.top_k(similarity_matrix, k=k)
    return indices


def similarity_from_labels(labels, vector_matrix, name='similarity_ground_truth'):
    """
    Computes similarity vectors for the ground truth, s=M*y
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param name: name for the operation
    :return: operation that computes the similarity vectors for the ground truth
    """
    # get the vectors for each label, y
    target_vectors = tf.gather(vector_matrix, labels)
    # s = y*M' -> dimensions: [batch_size, num_classes]
    return tf.matmul(target_vectors, vector_matrix, transpose_b=True, name=name)


def similarity_from_predictions(outputs, vector_matrix, name='similarity_predictions'):
    """
    Computes similarity vectors for the ground truth, s=M*y_
    :param outputs: L2-normalized outputs of the model. Dimensions: [batch_size, vector_dim]
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param name: name for the operation
    :return: operation that computes the similarity vectors for the ground truth
    """
    # s = y*M' -> dimensions: [batch_size, num_classes]
    return tf.matmul(outputs, vector_matrix, transpose_b=True, name=name)


def raw_cross_entropy(outputs, target, epsilon=1e-9, name='cross_entropy'):
    """
    Computes cross-entropy between outputs and target, without any normalization, as -sum[P(i)*log(Q(i))]
    :param outputs: outputs from the model. Tensor of shape [batch_size, n_dims].
    :param target: target distribution. Tensor of shape [batch_size, n_dims].
    :param epsilon: constant that ensures numerical stability of the log().
    :param name: name for the operation.
    :return: average cross-entropy for all the samples in the batch
    """
    cross_entropy_per_sample = -tf.reduce_sum(tf.mul(target, tf.log(tf.clip_by_value(outputs, epsilon, 1.0))), 1,
                                              name='%s_per_sample' % name)
    return tf.reduce_mean(cross_entropy_per_sample, 0, name=name)


def embedded_vectors_from_labels(labels, vector_matrix, name='ground_truth'):
    """
    Returns the embedded vector for each label
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param name: name for the operation
    :return: operation that returns the embedded vectors for each label in the ground truth
    """
    return tf.gather(vector_matrix, labels, name=name)


def generate_label_smoothing(vectors_file, name):
    # Load vectors. M has dims [num_classes, vector_dims]
    with open(vectors_file, 'rb') as f:
        vector_matrix = pickle.load(f, encoding='latin1')

    # Normalize vectors
    for class_index in range(np.shape(vector_matrix)[0]):
        vector_matrix[class_index, :] /= np.linalg.norm(vector_matrix[class_index, :])

    # Compute all similarities for all classes as M*M^T
    ground_truth_matrix = np.dot(vector_matrix, vector_matrix.T)

    # Normalize rows so that they sum 1
    for row in range(ground_truth_matrix.shape[0]):
        ground_truth_matrix[row, :] /= np.sum(ground_truth_matrix[row, :])

    return tf.get_variable(name, [ground_truth_matrix.shape[0], ground_truth_matrix.shape[1]],
                           initializer=tf.constant_initializer(ground_truth_matrix),
                           trainable=False)


def triplet_loss(outputs, labels, vector_matrix, m=0.5, batch_size=None, name='triplet_loss'):
    """
    Computes the triplet loss of the output with respect to the ground truth
    :param outputs: L2-normalized outputs of the model. Dimensions: [batch_size, vector_dim]
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param m: margin constant in the loss
    :param batch_size: batch size
    :param name: name for the loss operation
    :return: loss
    """
    if batch_size is None:
        batch_size = FLAGS.batch_size

    # [batch_size, vector_dim] * [vector_dim, num_classes] = [batch_size, num_classes]
    similarity_matrix = tf.matmul(outputs, vector_matrix, transpose_b=True)

    # Get the similarity w.r.t. the target vectors
    target_vectors = tf.gather(vector_matrix, labels)  # dims: [batch_size, vector_dims]
    similarity_wrt_gt = tf.reduce_sum(tf.mul(target_vectors, outputs), 1)
    # similarity_wrt_gt = tf.diag_part(tf.matmul(target_vectors, outputs, transpose_b=True))

    # Sort similarities, without using the target class (max similarity is 1)
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = vector_matrix.get_shape()[0].value
    mask = tf.sparse_to_dense(concated, [batch_size, num_classes], 0.0, 1.0)
    _, hard_negative_indices = tf.nn.top_k(tf.mul(similarity_matrix, mask), k=1)

    # Get the similarity w.r.t. the hard negatives
    hard_negative_vectors = tf.squeeze(tf.gather(vector_matrix, hard_negative_indices))
    similarity_wrt_hard_negatives = tf.reduce_sum(tf.mul(hard_negative_vectors, outputs), 1)

    # Compute triplet loss
    loss_per_sample = tf.maximum(0., m - similarity_wrt_gt + similarity_wrt_hard_negatives)
    return tf.reduce_mean(loss_per_sample, 0, name=name)


def hinge_rank_loss(outputs, labels, vector_matrix, m=0.1, max_iter=None, batch_size=None, name='hinge_rank_loss'):
    """
    Computes the hinge rank loss of the output with respect to the ground truth, based on
        Frome, A., et al. DeViSE: A deep visual-semantic embedding model. NIPS 2013.
    :param outputs: L2-normalized outputs of the model. Dimensions: [batch_size, vector_dim]
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param m: margin constant in the loss
    :param max_iter: truncates the sum even if there are samples with zero loss. None does not impose any limitation.
    :param batch_size: batch size
    :param name: name for the loss operation
    :return: loss
    """
    def cond(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter):
        """Condition for the while loop: keep iterating until all samples have a loss greater than 0"""
        non_positive_loss = tf.less(tf.reduce_mean(tf.cast(end_flag_per_sample, tf.int32)), 1)
        non_max_iter = tf.less(iteration_index, max_iter)
        return tf.logical_and(non_positive_loss, non_max_iter)

    def body(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter):
        """Sums triplets to the loss until cond returns False"""
        # Get the similarity w.r.t. the target vectors
        positive_vectors = tf.gather(vector_matrix, labels)  # dims: [batch_size, vector_dims]
        s_hat_i = tf.reduce_sum(tf.mul(positive_vectors, outputs), 1)

        # Get the similarity w.r.t. a random negative sample
        current_negative_indices = tf.transpose(tf.gather(tf.transpose(shuffled_labels), iteration_index))
        negative_vectors = tf.gather(vector_matrix, current_negative_indices)
        s_hat_j = tf.reduce_sum(tf.mul(negative_vectors, outputs), 1)

        # Compute new triplet losses only for samples with zero loss and pos_class != neg_class
        loss_inc = tf.maximum(0., m - s_hat_i + s_hat_j)
        valid_samples = tf.cast(tf.not_equal(current_negative_indices, labels), tf.float32)
        ret_loss = tf.add(loss,
                          tf.mul(valid_samples,
                                 tf.mul((1. - tf.cast(end_flag_per_sample, tf.float32)), loss_inc)))

        # Update end_flag_per_sample and iteration_index
        ret_end_flag_per_sample = tf.cast(tf.greater(loss, 0.), tf.int32)

        return ret_loss, outputs, labels, vector_matrix, m, ret_end_flag_per_sample, shuffled_labels,\
            iteration_index+1, max_iter

    if batch_size is None:
        batch_size = FLAGS.batch_size

    n_classes = vector_matrix.get_shape()[0].value

    if max_iter is None:
        max_iter = n_classes

    end_flag_per_sample = tf.constant([0] * batch_size, tf.int32)
    loss = tf.constant(0., dtype=tf.float32, shape=[batch_size], name='loss_sum')
    iteration_index = tf.constant(0, dtype=tf.int32)
    sorted_labels = tf.constant([i for i in range(n_classes)], dtype=tf.int32)
    shuffled_labels = tf.pack([tf.random_shuffle(sorted_labels) for _ in range(batch_size)])

    hinge_loss, _, _, _, _, _, _, _, _ = tf.while_loop(cond, body,
                                                       [loss,
                                                        outputs,
                                                        labels,
                                                        vector_matrix,
                                                        m,
                                                        end_flag_per_sample,
                                                        shuffled_labels,
                                                        iteration_index,
                                                        max_iter],
                                                       parallel_iterations=batch_size)

    return tf.reduce_mean(hinge_loss, name=name)


def weighted_hinge_rank_loss(outputs, labels, vector_matrix, m=0.1, max_iter=None, batch_size=None, version=1,
                             name='weighted_hinge_rank_loss'):
    """
    Hinge rank loss that weights each triplet depending on the similarity between the target and negative classes
    :param outputs: L2-normalized outputs of the model. Dimensions: [batch_size, vector_dim]
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing L2-normalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param m: margin constant in the loss. Not used if version == 3.
    :param max_iter: truncates the sum even if there are samples with zero loss. None does not impose any limitation.
    :param batch_size: batch size
    :param version: chooses which weighted hinge rank loss will be used:
        1: max(0., (1. - s_ij) * (m - s_hat_i + s_hat_j))
        2: max(0., (m - s_hat_i + (1. - s_ij) * s_hat_j))
        3: max(0., ((1. - s_ij) - s_hat_i + s_hat_j))
    :param name: name for the loss operation
    :return: loss
    """

    def cond(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter,
             similarity_matrix):
        """Condition for the while loop: keep iterating until all samples have a loss greater than 0"""
        non_positive_loss = tf.less(tf.reduce_mean(tf.cast(end_flag_per_sample, tf.int32)), 1)
        non_max_iter = tf.less(iteration_index, max_iter)
        return tf.logical_and(non_positive_loss, non_max_iter)

    def body(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter,
             similarity_matrix):
        """Sums triplets to the loss until cond returns False"""
        # Get the similarity w.r.t. the target vectors, s_hat_i (where i is the label)
        positive_vectors = tf.gather(vector_matrix, labels)  # dims: [batch_size, vector_dims]
        s_hat_i = tf.reduce_sum(tf.mul(positive_vectors, outputs), 1)

        # Get the similarity w.r.t. a random negative sample, s_hat_j
        current_negative_indices = tf.transpose(tf.gather(tf.transpose(shuffled_labels), iteration_index))
        negative_vectors = tf.gather(vector_matrix, current_negative_indices)  # index of the negative classes, j
        s_hat_j = tf.reduce_sum(tf.mul(negative_vectors, outputs), 1)
        
        # Get the similarity between target and negative classes, s_ij
        # Use sparse tensors because gather_nd gradients are not yet implemented
        # s_ij = tf.gather_nd(similarity_matrix, tf.pack([labels, current_negative_indices], axis=1))
        s_ij_mask = tf.sparse_to_dense(tf.pack([labels, current_negative_indices], axis=1),
                                       similarity_matrix.get_shape(),
                                       sparse_values=1., default_value=0., validate_indices=False)
        similarity_matrix_masked = tf.mul(similarity_matrix, s_ij_mask)
        s_ij = tf.reduce_sum(tf.gather(similarity_matrix_masked, labels), 1)

        # Compute loss for the new triplet depending on the version of the loss
        if version == 1:
            loss_inc = tf.maximum(0., (1. - s_ij) * (m - s_hat_i + s_hat_j))
        elif version == 2:
            loss_inc = tf.maximum(0., (m - s_hat_i + (1. - s_ij) * s_hat_j))
        elif version == 3:
            loss_inc = tf.maximum(0., ((1. - s_ij) - s_hat_i + s_hat_j))
        else:
            raise ValueError('Non-valid loss version (%d). It should be 1, 2 or 3.' % version)

        # Accumulate loss only for valid samples, i.e. i != j and loss == 0
        valid_samples = tf.cast(tf.not_equal(current_negative_indices, labels), tf.float32)
        ret_loss = tf.add(loss,
                          tf.mul(valid_samples,
                                 tf.mul((1. - tf.cast(end_flag_per_sample, tf.float32)), loss_inc)))

        # Update end_flag_per_sample and iteration_index
        ret_end_flag_per_sample = tf.cast(tf.greater(loss, 0.), tf.int32)

        return ret_loss, outputs, labels, vector_matrix, m, ret_end_flag_per_sample, shuffled_labels, \
               iteration_index + 1, max_iter, similarity_matrix

    if batch_size is None:
        batch_size = FLAGS.batch_size

    n_classes = vector_matrix.get_shape()[0].value

    if max_iter is None:
        max_iter = n_classes

    similarity_matrix = tf.matmul(vector_matrix, vector_matrix, transpose_b=True)  # [num_classes, num_classes]

    end_flag_per_sample = tf.constant([0] * batch_size, tf.int32)
    loss = tf.constant(0., dtype=tf.float32, shape=[batch_size], name='loss_sum')
    iteration_index = tf.constant(0, dtype=tf.int32)
    sorted_labels = tf.constant([i for i in range(n_classes)], dtype=tf.int32)
    shuffled_labels = tf.pack([tf.random_shuffle(sorted_labels) for _ in range(batch_size)])

    hinge_loss, _, _, _, _, _, _, _, _, _ = tf.while_loop(cond, body,
                                                          [loss,
                                                           outputs,
                                                           labels,
                                                           vector_matrix,
                                                           m,
                                                           end_flag_per_sample,
                                                           shuffled_labels,
                                                           iteration_index,
                                                           max_iter,
                                                           similarity_matrix],
                                                          parallel_iterations=batch_size)

    return tf.reduce_mean(hinge_loss, name=name)


def l2_hinge_rank_loss(outputs, labels, vector_matrix, m=0., max_iter=None, batch_size=None, name='hinge_rank_loss'):
    """
    :param outputs: unnormalized outputs of the model. Dimensions: [batch_size, vector_dim]
    :param labels: ground truth class index for each sample
    :param vector_matrix: ground truth matrix, containing unnormalized vectors for all the classes.
                          Dimensions: [num_classes, vector_dim]
    :param m: margin constant in the loss
    :param max_iter: truncates the sum even if there are samples with zero loss. None does not impose any limitation.
    :param batch_size: batch size
    :param name: name for the loss operation
    :return: loss
    """
    def cond(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter):
        """Condition for the while loop: keep iterating until all samples have a loss greater than 0"""
        non_positive_loss = tf.less(tf.reduce_mean(tf.cast(end_flag_per_sample, tf.int32)), 1)
        non_max_iter = tf.less(iteration_index, max_iter)
        return tf.logical_and(non_positive_loss, non_max_iter)

    def body(loss, outputs, labels, vector_matrix, m, end_flag_per_sample, shuffled_labels, iteration_index, max_iter):
        """Sums triplets to the loss until cond returns False"""
        # Get the distance w.r.t. the target vectors
        positive_vectors = tf.gather(vector_matrix, labels)  # dims: [batch_size, vector_dims]
        d_hat_i = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(positive_vectors, outputs)), reduction_indices=1))

        # Get the distance w.r.t. a random negative sample
        current_negative_indices = tf.transpose(tf.gather(tf.transpose(shuffled_labels), iteration_index))
        negative_vectors = tf.gather(vector_matrix, current_negative_indices)
        d_hat_j = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(negative_vectors, outputs)), reduction_indices=1))

        # Compute new triplet losses only for samples with zero loss and pos_class != neg_class
        loss_inc = tf.maximum(0., m + d_hat_i + d_hat_j)
        valid_samples = tf.cast(tf.not_equal(current_negative_indices, labels), tf.float32)
        ret_loss = tf.add(loss,
                          tf.mul(valid_samples,
                                 tf.mul((1. - tf.cast(end_flag_per_sample, tf.float32)), loss_inc)))

        # Update end_flag_per_sample and iteration_index
        ret_end_flag_per_sample = tf.cast(tf.greater(loss, 0.), tf.int32)

        return ret_loss, outputs, labels, vector_matrix, m, ret_end_flag_per_sample, shuffled_labels,\
            iteration_index+1, max_iter

    if batch_size is None:
        batch_size = FLAGS.batch_size

    n_classes = vector_matrix.get_shape()[0].value

    if max_iter is None:
        max_iter = n_classes

    end_flag_per_sample = tf.constant([0] * batch_size, tf.int32)
    loss = tf.constant(0., dtype=tf.float32, shape=[batch_size], name='loss_sum')
    iteration_index = tf.constant(0, dtype=tf.int32)
    sorted_labels = tf.constant([i for i in range(n_classes)], dtype=tf.int32)
    shuffled_labels = tf.pack([tf.random_shuffle(sorted_labels) for _ in range(batch_size)])

    hinge_loss, _, _, _, _, _, _, _, _ = tf.while_loop(cond, body,
                                                       [loss,
                                                        outputs,
                                                        labels,
                                                        vector_matrix,
                                                        m,
                                                        end_flag_per_sample,
                                                        shuffled_labels,
                                                        iteration_index,
                                                        max_iter],
                                                       parallel_iterations=batch_size)

    return tf.reduce_mean(hinge_loss, name=name)
