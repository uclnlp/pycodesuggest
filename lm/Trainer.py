import tensorflow as tf
import numpy as np


def get_evals(evals, model):
    for c, m in model.final_state[0] if model.is_attention_model else model.final_state:
        evals.append(c)
        evals.append(m)

    if model.is_attention_model:
        evals.extend(model.final_state[1] + model.final_state[2] + model.final_state[3] + model.final_state[4])
        evals.append(model.final_state[5])

    return evals


def get_initial_state(model):
    state = []
    att_states = None
    att_ids = None
    att_counts = None
    for c, m in model.initial_state[0] if model.is_attention_model else model.initial_state:
        state.append((c.eval(), m.eval()))
    if model.is_attention_model:
        att_states = [s.eval() for s in list(model.initial_state[1])]
        att_ids = [s.eval() for s in list(model.initial_state[2])]
        att_counts = [s.eval() for s in list(model.initial_state[4])]

    return state, att_states, att_ids, att_counts


def construct_feed_dict(model, seq_batch, state, att_states, att_ids, att_counts):
    input_data, targets, masks, identifier_usage, actual_lengths = seq_batch

    feed_dict = {
        model.input_data: input_data,
        model.targets: targets,
        model.actual_lengths: actual_lengths
    }

    if model.is_attention_model:
        feed_dict[model.masks] = masks

    for i, (c, m) in enumerate(model.initial_state[0]) if model.is_attention_model else enumerate(model.initial_state):
        feed_dict[c], feed_dict[m] = state[i]

    if model.is_attention_model:
        for i in range(len(model.initial_state[1])):
            feed_dict[model.initial_state[1][i]] = att_states[i]
            feed_dict[model.initial_state[2][i]] = att_ids[i]
            feed_dict[model.initial_state[4][i]] = att_counts[i]

    return feed_dict, identifier_usage


def extract_results(results, evals, num_evals, model):
    state_start = num_evals
    state_end = state_start + len(model.final_state[0])*2 if model.is_attention_model else len(evals)

    state_flat = results[state_start:state_end]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

    num_att_states = len(model.final_state[1]) if model.is_attention_model else 0
    att_states = results[state_end:state_end + num_att_states] if model.is_attention_model else None
    att_ids = results[state_end+num_att_states:state_end + num_att_states*2] if model.is_attention_model else None
    alpha_states = results[state_end+num_att_states*2:state_end+num_att_states*3] if model.is_attention_model else None
    att_counts = results[state_end+num_att_states*3:state_end+num_att_states*4]
    lambda_state = results[-1] if model.is_attention_model else None

    return results[0:num_evals], state, att_states, att_ids, alpha_states, att_counts, lambda_state


def iterate_batch(evals, model, batcher, session, hook):
    num_evals = len(evals)
    evals = get_evals(evals, model)

    for batch in batcher:
        state, att_states, att_ids, att_counts = get_initial_state(model)

        for seq_batch in batcher.sequence_iterator(batch):

            feed_dict, identifier_usage = construct_feed_dict(model, seq_batch, state, att_states, att_ids, att_counts)

            results = session.run(evals, feed_dict=feed_dict)

            results, state, att_states, att_ids, alpha_states, att_counts, lambda_state \
                = extract_results(results, evals, num_evals, model)

            hook(results, feed_dict[model.actual_lengths], identifier_usage)


def print_feed_dict(feed_dict):
    keys = sorted(feed_dict.keys(), key=lambda k: k.name)
    for key in keys:
        print("%s : %s" % (key.name, key._shape))
        print(feed_dict[key])


class Trainer(object):

    """
    Object representing a TensorFlow trainer.
    """

    class Tracker:
        def __init__(self):
            self.total_loss = 0.0
            self.processed = 0.0
            self.iteration = 1

    def __init__(self, optimizer, epochs, hooks, model, minimization_op=None):
        self.loss = None
        self.optimizer = optimizer
        self.epochs = epochs
        self.hooks = hooks
        self.minimization_op = minimization_op
        self.model = model

    def __call__(self, batcher, loss, session, learning_rate, lr_decay, load_func=None):
        self.loss = loss
        minimization_op = self.minimization_op or self.optimizer.minimize(loss)

        init = tf.initialize_all_variables()
        session.run(init)

        if load_func is not None:
            load_func()

        for hook in self.hooks:
            hook(session, 0, 0, self.model.logits, 0, 0)

        start_decaying = self.epochs // 4

        for epoch in range(1, self.epochs+1):
            epoch_tracker = Trainer.Tracker()

            def iter_hook(results, actual_lengths, _):
                current_loss = sum(results[1])
                epoch_tracker.total_loss += current_loss
                epoch_tracker.processed += sum(actual_lengths)
                epoch_tracker.iteration += 1

                for hook in self.hooks:
                    hook(session, epoch, epoch_tracker.iteration, self.model, current_loss, epoch_tracker.processed)

            lr = learning_rate if epoch < start_decaying else learning_rate * lr_decay
            self.model.assign_lr(session, lr)

            evals = [minimization_op, loss]
            try:
                iterate_batch(evals, self.model, batcher, session, iter_hook)
            except KeyboardInterrupt:
                print("Keyboard interrupt detected, exiting training loop")
                break

            print("Training perplexity %f" % np.exp((epoch_tracker.total_loss/epoch_tracker.processed)))

            # calling post-epoch hooks
            for hook in self.hooks:
                hook(session, epoch, 0, self.model.logits, 0, epoch_tracker.processed)
