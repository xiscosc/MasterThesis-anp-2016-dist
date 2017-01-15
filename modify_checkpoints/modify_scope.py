import tensorflow as tf


def assign_to_new_scope(new_scope, old_scope, var_list=None):
    """
    Allows to change the scope of a pre-trained model. Returns a list of operations for copying values of all
    variables of the kind old_scope/* to all variables named new_scope/*
    :param new_scope:
    :param old_scope:
    :param var_list: list of variables to consider. None uses all variables in the graph under old_scope.
    :return: list of assign operations that need to be run
    """
    if var_list is None:
        all_vars = tf.all_variables()
    else:
        all_vars = var_list

    # Create dictionary with the original variables
    old_vars = {v.name.split('/', 1)[-1]: v for v in all_vars if v.name.startswith('%s/' % old_scope)}

    # Assign variables values under the new scope
    assign_ops = []
    for v in tf.all_variables():
        scope, var_name = v.name.split('/', 1)
        if scope == new_scope and var_name in old_vars:
            assign_ops.append(v.assign(old_vars[var_name]))
    return assign_ops
