import collections
import six
import tensorflow as tf


def is_sequence(seq):
    return (isinstance(seq, collections.Sequence) and
            not isinstance(seq, six.string_types))


def yield_unpacked_state(state):
    for s in state:
        if is_sequence(s):
            for si in yield_unpacked_state(s):
                yield si
        else:
            yield s


def unpacked_state(state):
    if not is_sequence(state):
        raise TypeError("state must be a sequence")
    return list(yield_unpacked_state(state))


def sequence_like(instance, args):
    try:
        assert isinstance(instance, tuple)
        assert isinstance(instance._fields, collections.Sequence)
        assert all(isinstance(f, six.string_types) for f in instance._fields)
        # This is a namedtuple
        return type(instance)(*args)
    except (AssertionError, AttributeError):
        # Not a namedtuple
        return type(instance)(args)


def packed_state_with_indices(structure, flat, index):
    packed = []
    for s in structure:
        if is_sequence(s):
            new_index, child = packed_state_with_indices(s, flat, index)
            packed.append(sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def packed_state(structure, state):
    if not is_sequence(structure):
        raise TypeError("structure must be a sequence")
    if not is_sequence(state):
        raise TypeError("state must be a sequence")

    flat_structure = unpacked_state(structure)
    if len(flat_structure) != len(state):
        raise ValueError(
            "Internal error: Could not pack state.  Structure had %d elements, but "
            "state had %d elements.  Structure: %s, state: %s."
            % (len(flat_structure), len(state), structure, state))

    (_, packed) = packed_state_with_indices(structure, state, 0)
    return sequence_like(structure, packed)


def shift_tensor(t):
    """
    Shift a tensor along its first dimension
    """

    shape = tf.shape(t)
    left_start = tf.concat(0, [[1], tf.zeros([tf.shape(shape)[0]-1], tf.int32)])
    left_shape = tf.concat(0, [[shape[0]-1], shape[1:]])
    zeros = tf.concat(0, [[1], shape[1:]])
    t = tf.slice(t, left_start, left_shape)
    t = tf.concat(0, [t, tf.zeros(zeros, tf.float32)])
    return t


def replace_matrix_row(matrix, index, row_vector, row_size=None):
    """
    Replace the row index in matrix with row_vector
    """

    row_size = row_size or tf.shape(matrix)[0]

    def shift_and_append():
        left = tf.slice(matrix, [1, 0], [-1, -1])
        return tf.concat(0, [left, tf.expand_dims(row_vector, 0)])

    def replace():
        left = tf.slice(matrix, [0, 0], [index, -1])
        right = tf.slice(matrix, [index + 1, 0], [row_size - index - 1, -1])
        return tf.concat(0, [left, tf.expand_dims(row_vector, 0), right])

    return tf.cond(index >= row_size,
                   shift_and_append,
                   replace)


def process(_, current):
    count = tf.cast(current[0], tf.int32)
    current = tf.slice(current, [1], [-1])
    max = tf.shape(current)[0]
    sm = tf.expand_dims(tf.slice(current, [max - count], [-1]), 0)
    sm = tf.nn.softmax(sm)
    return tf.concat(0, [tf.zeros([max-count]), tf.squeeze(sm, [0])])


def jagged_softmax(input, counts):
    counts = tf.cast(counts, tf.float32)
    inputs = tf.concat(1, [counts, input])
    output = tf.scan(process, inputs, tf.zeros([0]), back_prop=False)
    return output


def tile_vector(vector, number):
    return tf.reshape(tf.tile(tf.expand_dims(vector,1), [1, number]).eval(), [-1])


def sparse_transpose(sp_input):
    transposed_indices = tf.reverse(tf.cast(sp_input.indices, tf.int32), [False, True])
    transposed_values = sp_input.values
    transposed_shape = tf.reverse(tf.cast(sp_input.shape, tf.int32), [True])
    sp_output = tf.SparseTensor(tf.cast(transposed_indices, tf.int64), transposed_values, tf.cast(transposed_shape, tf.int64))
    sp_output = tf.sparse_reorder(sp_output)
    return sp_output


def cross_entropy(labels, predict, batch_size, vocab_size):
    indices = labels + (tf.range(batch_size) * vocab_size)
    predict_flat = tf.reshape(predict, [-1])
    gathered = tf.gather(predict_flat, indices)
    ce = -tf.log(gathered + 1e-10)
    return ce


def cross_entropy_from_indices(labels, indices, probabilities, batch_size, size):
    indices = tf.cast(indices, tf.int32)
    targets = tf.tile(tf.expand_dims(labels, 1), [1, size])
    selection = tf.select(tf.equal(indices, targets), probabilities, tf.zeros([batch_size, size]))
    ce = -tf.log(tf.reduce_sum(selection, 1) + 1e-10)
    return ce

