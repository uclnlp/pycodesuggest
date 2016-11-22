import tensorflow as tf

import tfutils

'''
Note: This code was adapted from the Tensorflow source
'''


def dynamic_attention_rnn(cell, inputs, attn_length, num_tasks, batch_size,
                          sequence_length=None, initial_state=None,
                          dtype=None, parallel_iterations=None, swap_memory=False, scope=None):

    parallel_iterations = parallel_iterations or 32

    sequence_length = tf.to_int32(sequence_length)
    sequence_length = tf.identity(sequence_length, name="sequence_length")  # Just to find it in the graph.

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        (outputs, alphas, attn_ids, lmbdas, final_state) = _dynamic_attention_rnn_loop(
            cell, inputs, state, parallel_iterations=parallel_iterations,
            swap_memory=swap_memory, sequence_length=sequence_length, attn_length=attn_length,
            num_tasks=num_tasks, batch_size=batch_size)

        # outputs = (steps, batch, size)
        # alphas = (steps, batch, tasks, K)
        # attn_ids = (steps, batch, tasks, K)
        # lmbdas = (steps, batch, tasks)

        return outputs, alphas, attn_ids, lmbdas, final_state


def _dynamic_attention_rnn_loop(cell, inputs, initial_state, parallel_iterations,
                                swap_memory, sequence_length, attn_length, num_tasks,
                                batch_size):
    state = initial_state

    # Construct an initial output
    input_shape = tf.shape(inputs)
    (time_steps, _, _) = tf.unpack(input_shape, 3)

    inputs_got_shape = inputs.get_shape().with_rank(3)
    (const_time_steps, const_batch_size, const_depth) = inputs_got_shape.as_list()

    # Prepare dynamic conditional copying of state & output
    zero_output = tf.zeros(tf.pack([batch_size, cell.output_size]), inputs.dtype)
    zero_attn_ids = zero_alpha = tf.zeros([batch_size, num_tasks-1, attn_length], inputs.dtype)
    zero_lmbdas = tf.zeros([batch_size, num_tasks], tf.float32)

    time = tf.constant(0, dtype=tf.int32, name="time")

    state_size = cell.state_size

    state = tfutils.unpacked_state(state)

    with tf.op_scope([], "dynamic_rnn") as scope:
        base_name = scope

    def create_ta(name, dtype=None):
        dtype = dtype or inputs.dtype
        return tf.TensorArray(dtype=dtype, size=time_steps, tensor_array_name=base_name + name)

    output_ta = create_ta("output")
    alpha_ta = create_ta("alpha", tf.float32)
    attn_ids_ta = create_ta("attn_ids")
    lmbda_ta = create_ta("lmbdas", tf.float32)
    input_ta = create_ta("input")

    input_ta = input_ta.unpack(inputs)

    def _time_step(time, output_ta_t, alpha_ta_t, attn_ids_ta_t, lmbda_ta_t, *state):
        input_t = input_ta.read(time)
        # Restore some shape information
        input_t.set_shape([const_batch_size, const_depth])

        # Pack state back up for use by cell
        state = tfutils.packed_state(structure=state_size, state=state)

        call_cell = lambda: cell(input_t, state)

        (output, alpha, attn_ids, lmbdas, new_state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            zero_output=zero_output,
            zero_alpha=zero_alpha,
            zero_attn_ids=zero_attn_ids,
            zero_lmbdas=zero_lmbdas,
            state=state,
            call_cell=call_cell,
            state_size=state_size,
        )

        # Pack state if using state tuples
        new_state = tuple(tfutils.unpacked_state(new_state))

        output_ta_t = output_ta_t.write(time, output)
        alpha_ta_t = alpha_ta_t.write(time, alpha)
        attn_ids_ta_t = attn_ids_ta_t.write(time, attn_ids)
        lmbda_ta_t = lmbda_ta_t.write(time, lmbdas)

        return (time + 1, output_ta_t, alpha_ta_t, attn_ids_ta_t, lmbda_ta_t) + new_state

    final_loop_vars = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, alpha_ta, attn_ids_ta, lmbda_ta) + tuple(state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    (output_final_ta, alpha_final_ta, attn_ids_final_ta, lmbda_final_ta, final_state) = \
        (final_loop_vars[1], final_loop_vars[2], final_loop_vars[3], final_loop_vars[4], final_loop_vars[5:])

    final_outputs = output_final_ta.pack()
    final_alphas = alpha_final_ta.pack()
    final_attn_ids = attn_ids_final_ta.pack()
    final_lmbdas = lmbda_final_ta.pack()
    # Restore some shape information
    final_outputs.set_shape([const_time_steps, const_batch_size, cell.output_size])
    final_alphas.set_shape([const_time_steps, const_batch_size, num_tasks-1, attn_length])
    final_attn_ids.set_shape([const_time_steps, const_batch_size, num_tasks-1, attn_length])
    final_lmbdas.set_shape([const_time_steps, const_batch_size, num_tasks])

    # Unpack final state if not using state tuples.
    final_state = tfutils.packed_state(structure=cell.state_size, state=final_state)

    return final_outputs, final_alphas, final_attn_ids, final_lmbdas, final_state


def _rnn_step(time, sequence_length, zero_output, zero_alpha, zero_attn_ids, zero_lmbdas, state, call_cell, state_size):
    # Convert state to a list for ease of use
    state = list(tfutils.unpacked_state(state))
    state_shape = [s.get_shape() for s in state]

    def _copy_some_through(new_output, new_alpha, new_attn_ids, new_lmbdas, new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.

        # Alpha needs to be (batch, tasks, k)
        copy_cond = (time >= sequence_length)
        return ([tf.select(copy_cond, zero_output, new_output),
                 tf.select(copy_cond, zero_alpha, new_alpha), # (batch, tasks, k)
                 tf.select(copy_cond, zero_attn_ids, new_attn_ids),
                 tf.select(copy_cond, zero_lmbdas, new_lmbdas)] +
                [tf.select(copy_cond, old_s, new_s)
                 for (old_s, new_s) in zip(state, new_state)])

    new_output, new_alpha, new_attn_ids, new_lmbdas, new_state = call_cell()
    new_state = list(tfutils.unpacked_state(new_state))

    final_output_and_state = _copy_some_through(new_output, new_alpha, new_attn_ids, new_lmbdas, new_state)

    (final_output, final_alpha, final_attn_ids, final_lmbdas, final_state) = (
        final_output_and_state[0], final_output_and_state[1], final_output_and_state[2],
        final_output_and_state[3], final_output_and_state[4:])

    final_output.set_shape(zero_output.get_shape())
    final_alpha.set_shape(zero_alpha.get_shape())
    final_attn_ids.set_shape(zero_attn_ids.get_shape())
    final_lmbdas.set_shape(zero_lmbdas.get_shape())

    for final_state_i, state_shape_i in zip(final_state, state_shape):
        final_state_i.set_shape(state_shape_i)

    return (
        final_output,
        final_alpha,
        final_attn_ids,
        final_lmbdas,
        tfutils.packed_state(structure=state_size, state=final_state))


def attention_rnn(cell, inputs, num_steps, initial_state, batch_size, size, attn_length, num_tasks, sequence_length=None):
    '''

    :param cell: Cell takes input and state as input and returns output, alpha, attn_ids, lambda and new_state
    :param inputs: An tensor of size (batch x steps x size)
    :param attn_length:
    :param num_tasks:
    :param sequence_length:
    :param initial_state:
    :return:
    '''

    outputs = []
    alphas = []
    attn_ids = []
    lmbdas = []

    state = initial_state

    with tf.variable_scope("RNN"):
        time = tf.constant(0)
        for t in range(num_steps):
            if t > 0:
                tf.get_variable_scope().reuse_variables()

            (output, alpha, attn_id, lmbda, state) = cell(inputs[t, :, :], state)

            outputs.append(output)  # output = (batch, size)
            alphas.append(alpha)  # alpha = (tasks, batch, attn_length)
            attn_ids.append(attn_id)  # attn_ids = (tasks, batch, attn_length)
            lmbdas.append(lmbda)  # lmbdas = (batch, tasks)

            time += 1

        output_tensor = tf.pack(outputs)
        alpha_tensor = tf.pack(alphas)
        attn_id_tensor = tf.pack(attn_ids)
        lmbda_tensor = tf.pack(lmbdas)

    return output_tensor, alpha_tensor, attn_id_tensor, lmbda_tensor, state
