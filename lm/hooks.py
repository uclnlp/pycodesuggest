import time
from Trainer import iterate_batch, get_evals, get_initial_state, construct_feed_dict, extract_results
import tensorflow as tf
import numpy as np
import sys
import tfutils
from utils import attention_masks, save_model, load_model

PERPLEXITY_TRACE_TAG = "Perplexity"
LOSS_TRACE_TAG = "Loss"
SPEED_TRACE_TAG = "Speed"
ACCURACY_TRACE_TAG = "Accuracy"


class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        raise NotImplementedError


class TraceHook(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        self.title_placeholder = tf.placeholder(tf.string)
        self.value_placeholder = tf.placeholder(tf.float64)
        cur_summary = tf.scalar_summary(self.title_placeholder, self.value_placeholder)
        self.merged_summary_op = tf.merge_summary([cur_summary])

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        if self.summary_writer is not None:
            summary_str = sess.run(self.merged_summary_op, {
                self.title_placeholder: title,
                self.value_placeholder: value
            })
            self.summary_writer.add_summary(summary_str, current_step)


class LossHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval):
        super(LossHook, self).__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.acc_loss = 0

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        self.acc_loss += loss
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            loss = self.acc_loss / self.iteration_interval
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tLoss " + str(loss))
            self.update_summary(sess, iteration, LOSS_TRACE_TAG, loss)
            self.acc_loss = 0


class SpeedHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval, batch_size):
        super(SpeedHook, self).__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.batch_size = batch_size
        self.t0 = time.time()
        self.processed0 = 0
        self.counter = 1

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        if iteration == 0:
            self.counter = 1
            self.processed0 = 0

        if self.counter == self.iteration_interval:
            self.counter = 1
            diff = time.time() - self.t0
            speed = int((processed - self.processed0) / diff)
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tExamples/s " + str(speed))
            self.update_summary(sess, iteration, SPEED_TRACE_TAG, float(speed))
            self.t0 = time.time()
            self.processed0 = processed
        else:
            self.counter += 1


class AccuracyHook(TraceHook):
    def __init__(self, summary_writer, batcher, placeholders, at_every_epoch):
        super(AccuracyHook, self).__init__(summary_writer)
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            correct = 0
            for values in self.batcher:
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)
                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
                correct += sum(truth == predicted)
            acc = float(correct) / total
            self.update_summary(sess, iteration, ACCURACY_TRACE_TAG, acc)
            print("Epoch " + str(epoch) +
                  "\tAcc " + str(acc) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))


class SaveModelHook(Hook):
    def __init__(self, path, at_epoch, config, at_every_epoch=1):
        self.path = path + "/"
        self.at_epoch = at_epoch
        self.at_every_epoch = at_every_epoch
        self.config = config

        # fixme: don't save optimizer parameters
        # self.saver = tf.train.Saver(tf.all_variables())
        self.saver = tf.train.Saver(tf.trainable_variables())

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            print("Saving model...")
            save_model(self.saver, sess, self.path, model, self.config)


class LoadModelHook(Hook):
    def __init__(self, path, at_epoch, at_every_epoch=0):
        self.path = path + "/"
        self.at_epoch = at_epoch
        self.at_every_epoch = at_every_epoch
        self.saver = tf.train.Saver(tf.all_variables())

    def __call__(self, sess, epoch, iteration, model, loss, processed):
        if epoch == self.at_epoch:
            print("Loading model...")
            model = load_model(sess, self.path + "latest/")



class PerplexityHook(TraceHook):
    def __init__(self, summary_writer, model, batcher, at_every_epoch=1):
        super(PerplexityHook, self).__init__(summary_writer)
        labels = tf.reshape(model.targets, [-1])
        self.cross_entropy = tfutils.cross_entropy(labels, model.predict,
                                                   model.config.batch_size * model.config.seq_length,
                                                   model.config.vocab_size)
        self.mask = tf.sign(tf.abs(model.targets))
        self.mask = tf.cast(tf.reshape(self.mask, [-1]), tf.float32)
        self.cross_entropy *= self.mask  # Zero out entries where the target is 0 (padding)
        self.cost_op = tf.reduce_sum(self.cross_entropy)
        self.batcher = batcher
        self.at_every_epoch = at_every_epoch
        self.model = model

    def __call__(self, sess, epoch, iteration, _, __, ___):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            vals = {"cost": 0.0, "iters": 0.0}

            def iter_hook(results, actual_lengths, _):
                vals["cost"] += results[0]
                vals["iters"] += sum(actual_lengths)

            evals = [self.cost_op, self.cross_entropy, self.mask]
            iterate_batch(evals, self.model, self.batcher, sess, iter_hook)

            perplexity = np.exp(vals["cost"] / vals["iters"])
            self.update_summary(sess, epoch, PERPLEXITY_TRACE_TAG + (self.batcher.description or ""), perplexity)

            print("Epoch " + str(epoch) + "\t" + (self.batcher.description or "") +
                  "\tPerplexity " + str(perplexity))

            sys.stdout.flush()


# Pass the qualitative test set in to prime the samples
class GeneratorHook(Hook):
    def __init__(self, model, map, attns, at_every_epoch=1, sample_length=20):
        if model.seq_length != 1 or model.batch_size != 1:
            raise ValueError("Generator Hook only works with a model with a batch size and sequence length of 1")

        self.model = model
        self.attns = attns
        self.at_every_epoch = at_every_epoch
        self.prediction_op = tf.argmax(model.predict, 1)
        self.sample_length = sample_length
        self.map = map
        self.inverse_map = {v: k for k, v in map.items()}
        self.test_cases = [["class", "MyClass", ":", "§<indent>§", "def", "__init__", "(", "self", ",", "name", ",", "surname", ")", ":"],
                           ["filename", "=", "§OOV§", "\n", "with", "open"],
                           ["def", "§OOV§", "(", "§OOV§", ")", ":", "\n", "§<indent>§", "for"],
                           ["def", "§OOV§", "(", "§OOV§", ")", ":", "\n", "§<indent>§", "with", "open", "(", "§OOV§", ",", "'r'", ")", "as", "f", ":", "\n"],
                           ["with", "open", "(", "§OOV§", ",", "'r'", ")", "as", "f", ":", "\n"],
                           ["def", "§OOV§", "(", "name", ",", "surname", ")", ":", "\n", "§<indent>§", "fullname", "=", "name", "+", "' '", "+", "surname", "\n", "return"],
                           ["import", "numpy", "as", "np", "\n", "data", "=", "np", ".", "array", "(", "[", "§NUM§", ",", "§NUM§", ",", "§NUM§", ",", "§NUM§", ",", "]", ")", "\n", "np", "."]]

    def __call__(self, sess, epoch, iteration, model, loss, _):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            print("Generating %d samples from model" % len(self.test_cases))
            evals = get_evals([self.prediction_op], self.model)
            for testcase in self.test_cases:
                output = list(testcase)
                state, att_states, att_counts = get_initial_state(self.model)

                for i in range(len(testcase) + self.sample_length):
                    # TODO: Need to determine whether a generated token is a variable?
                    # Run it through the parser?
                    att_mask = attention_masks(self.attns, [0], 1)
                    data = (np.array([[self.map[output[i]]]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
                    feed_dict = construct_feed_dict(self.model, data, state, att_states, att_counts)

                    results = sess.run(evals, feed_dict)
                    results, state, att_states, att_counts, _, _ = extract_results(results, evals, 1, self.model)

                    output_token = self.inverse_map[results[0][0]]

                    if i >= len(testcase)-1:
                        output.append(output_token)

                print(" ".join(output))
                print("\n\n§§§§§§§§§§§§§§§§\n\n")


class TopKAccuracyHook(TraceHook):
    def __init__(self, summary_writer, model, batcher, ks, at_every_epoch=1):
        super(TopKAccuracyHook, self).__init__(summary_writer)
        self.ks = ks
        self.mask = tf.sign(tf.abs(model.targets))
        self.mask = tf.reshape(self.mask, [-1])
        self.at_every_epoch = at_every_epoch
        self.batcher = batcher
        self.model = model
        shaped_targets = tf.reshape(model.targets, [model.config.batch_size * model.config.seq_length])

        self.top_k_ops = [tf.cast(tf.nn.in_top_k(model.predict, shaped_targets, k), tf.int32)*self.mask
                          for k in ks]

    def __call__(self, sess, epoch, iteration, model, _, __):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            accuracy_vals_identifiers = {}
            accuracy_vals_nonidentifiers = {}
            accuracy = {}
            total_identifiers = 0.0
            total_non_identifiers = 0.0
            total = 0.0

            for k in self.ks:
                accuracy[k] = 0.0
                accuracy_vals_identifiers[k] = 0.0
                accuracy_vals_nonidentifiers[k] = 0.0

            def iter_hook(results, actual_lengths, identifier_usage):
                nonlocal total, total_identifiers, total_non_identifiers
                non_identifier_usage = 1-identifier_usage
                seq_len = self.model.config.seq_length
                batch_size = self.model.config.batch_size

                actual_lengths = actual_lengths.astype("int")
                for i in range(batch_size):
                    identifier_usage[i, actual_lengths[i]:] = 0
                    non_identifier_usage[i, actual_lengths[i]:] = 0

                total += sum(actual_lengths)
                total_identifiers += sum(sum(identifier_usage))
                total_non_identifiers += sum(sum(non_identifier_usage))

                for i, k in enumerate(self.ks):
                    accuracy[k] += sum(results[i])
                    accuracy_vals_identifiers[k] += sum(sum(np.reshape(results[i], [batch_size, seq_len]) * identifier_usage))
                    accuracy_vals_nonidentifiers[k] += sum(sum(np.reshape(results[i], [batch_size, seq_len]) * non_identifier_usage))

            iterate_batch(self.top_k_ops, self.model, self.batcher, sess, iter_hook)

            for k in self.ks:
                accuracy_k = accuracy[k]/total
                accuracy_identifiers = accuracy_vals_identifiers[k]/total_identifiers
                accuracy_non_identifiers = accuracy_vals_nonidentifiers[k]/total_non_identifiers

                print("Top %d Accuracy: %f" % (k, accuracy_k))
                print("No. Identifiers: %d\nTop %d Accuracy identifiers: %f" % (total_identifiers, k, accuracy_identifiers))
                print("No. Non-identifiers: %d\nTop %d Accuracy non-identifiers: %f" % (total_non_identifiers, k, accuracy_non_identifiers))

            sys.stdout.flush()
