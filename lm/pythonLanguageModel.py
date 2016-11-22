from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import astwalker
import evaluation
import prebatcher
import pyreader
import tensorflow as tf
import pickle
import os
import shutil
import tempfile
import datetime

from Trainer import Trainer
from batcher import QueuedSequenceBatcher, PreBatched
from tfrnn.hooks import SpeedHook, LossHook, SaveModelHook
from hooks import PerplexityHook

from tfrnn.util import save_model, load_model, load_variables
from utils import get_file_list, copy_temp_files, create_model

import sys

flags = tf.flags

flags.DEFINE_string("data_path", "/Users/avishkar/pythonRepos", "data path")
flags.DEFINE_boolean("train", False, "train the model")
flags.DEFINE_boolean("test", False, "test the model")
flags.DEFINE_boolean("preprocess", False, "Proprocess data")
flags.DEFINE_boolean("prebatch", False, "Pre-batch and split the data")
flags.DEFINE_boolean("vocab", False, "Generate vocab")
flags.DEFINE_string("list_file", "train_files.txt", "Name of the list file found in data_path")
flags.DEFINE_string("vocab_file", "mapping.map", "Name of the vocab file in data_path")
flags.DEFINE_string("output_file", None, "Name of the output file")
flags.DEFINE_boolean("debug", False, "Use debug config")
flags.DEFINE_integer("seq_length", 100, "Sequence length")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("num_partitions", None, "Data partitions")
flags.DEFINE_boolean("use_prebatched", True, "Use prebatched data")
flags.DEFINE_boolean("copy_temp", False, "Copy data to local temp directory")
flags.DEFINE_integer("oov_threshold", 10, "Out of vocabulary threshold")
flags.DEFINE_integer("epochs", 50, "Number of epochs to run")
flags.DEFINE_string("attention", None, "Use the attention model")
flags.DEFINE_string("attention_variant", None, "Variation of attention model to use. Possible values are: "
                                               "input, output")
flags.DEFINE_float("init_scale", 0.1, "Initialisation scale")
flags.DEFINE_integer("max_grad_norm", 5, "Maximum norm for gradients")
flags.DEFINE_integer("num_layers", 1, "Number of LSTM layers")
flags.DEFINE_integer("hidden_size", 200, "Size of hidden state")
flags.DEFINE_float("keep_prob", 0.9, "1 - probability of dropout of input")
flags.DEFINE_integer("num_samples", 100, "Number of samples for sampled softmax")
flags.DEFINE_integer("status_iterations", 1000, "Number of iterations before status messages are displayed")
flags.DEFINE_integer("max_attention", 10, "Maximum size of attention matrix")
flags.DEFINE_float("learning_rate", 1.0, "Gradient Descent Learning Rate")
flags.DEFINE_float("lr_decay", 0.9, "Learning rate decay factor")
flags.DEFINE_string("model_path", None, "Model parameters to load. If train=True, "
                                        "will continue training from these parameters. If test=True,"
                                        "will test using these model parameters")
flags.DEFINE_string("lambda_type", "state", "Method to calculate lambda, possible values are fixed, state, att, input."
                                            "state, att and input can be combined using + "
                                            "eg state+att for state and attention")
flags.DEFINE_string("save_path", "./out/model", "Path to save the final model")
flags.DEFINE_string("checkpoint_path", "./out/checkpoint", "Path to save intermediate checkpoints, every 5 epochs")
flags.DEFINE_string("events_path", "./out/save", "Path to save summary events")
flags.DEFINE_string("data_pattern", "all_{-type-}_data.dat", "Pattern for data files, use {-type-} as a placeholder"
                                                             "for train/valid/test and leave out the part extension")
flags.DEFINE_boolean("vocab_incl_ids", False, "Include normalised identifiers in the vocab")
flags.DEFINE_string("embedding_path", None, "Path to load embeddings from")
flags.DEFINE_boolean("embedding_trainable", True, "Flag indicating whether the embeddings are trainable or not."
                                                  "Useful if embeddings are provided with embedding_path which should"
                                                  "not be updated during training")

FLAGS = flags.FLAGS


def train(data_path, config):
    with tf.Graph().as_default(), tf.Session() as session:
        word_to_id_path = os.path.join(data_path, config.vocab_file)
        with open(word_to_id_path, "rb") as f:
            word_to_id = pickle.load(f)

        vocab_size = len(word_to_id)
        print("Vocab size: %d" % vocab_size)
        sys.stdout.flush()

        train_pattern = config.data_pattern.replace("{-type-}", "train") + ".part*"
        valid_pattern = config.data_pattern.replace("{-type-}", "valid") + ".part*"

        train_files = get_file_list(config, data_path, train_pattern, "train")
        valid_files = get_file_list(config, data_path, valid_pattern, "valid")

        if config.copy_temp:
            temp_dir = tempfile.mkdtemp()
            print("Copying data files to %s" % temp_dir)
            train_files = copy_temp_files(train_files, temp_dir)
            valid_files = copy_temp_files(valid_files, temp_dir)

        config.vocab_size = vocab_size

        train_batcher = PreBatched(train_files, config.batch_size, description="train") if config.use_prebatched \
            else QueuedSequenceBatcher(train_files, config.seq_length, config.batch_size, description="train",
                                       attns=config.attention)
        valid_batcher = PreBatched(valid_files, config.batch_size, description="valid") if config.use_prebatched \
            else QueuedSequenceBatcher(valid_files, config.seq_length, config.batch_size, description="valid",
                                       attns=config.attention)

        t0 = datetime.datetime.now()
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = create_model(config, True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = create_model(config, False)

        summary_writer = tf.train.SummaryWriter(config.events_path, graph=session.graph)
        valid_perplexity = PerplexityHook(summary_writer, mvalid, valid_batcher)

        hooks = [
            SpeedHook(summary_writer, config.status_iterations, config.batch_size),
            LossHook(summary_writer, config.status_iterations),
            valid_perplexity,
            SaveModelHook(config.checkpoint_path, 1, config.__dict__, 5)
        ]
        t1 = datetime.datetime.now()
        print("Building models took: %s" % (t1 - t0))

        def load_func():
            if config.model_path is not None:
                load_model(session, config.model_path)
                print("Continuing training from model: %s" % config.model_path)
            if config.embedding_path is not None:
                load_variables(session, os.path.join(config.embedding_path, "embedding.tf"),
                               [m.embedding_variable])
                print("Loading embedding vectors from: %s" % config.embedding_path)

        trainer = Trainer(m.optimizer, config.epochs, hooks, m, m.train_op)
        trainer(train_batcher, m.loss, session, config.learning_rate, config.lr_decay, load_func)

        saver = tf.train.Saver(tf.trainable_variables())
        embedding_saver = tf.train.Saver([m.embedding_variable])
        print("Saving model...")
        out_path = save_model(saver, session, config.save_path, m.predict, config.__dict__)
        embedding_saver.save(session, os.path.join(out_path, "embedding.tf"))

        if config.copy_temp:
            shutil.rmtree(temp_dir)


def test2(data_path, config):
    evaluation.eval(data_path, config)


def preprocess(data_path, config):
    if config.output_file is None:
        print("Output file parameter needed, aborting...")
        sys.exit()

    word_to_id_path = os.path.join(data_path, config.vocab_file)
    with open(word_to_id_path, "rb") as f:
        word_to_id = pickle.load(f)

    list_file = os.path.join(data_path, config.list_file)
    data = pyreader.get_data(data_path, list_file, config.seq_length, word_to_id)
    write_partitions(pyreader.partition_data(data, config.num_partitions),
                     os.path.join(data_path, config.output_file))


def write_partitions(partitions, filename_base):
    print("File: %s\nNumber of partitions: %d" % (filename_base, len(partitions)))

    counter = 0
    for partition_key in partitions:
        suffix = ".part" + str(counter)
        with open(filename_base + suffix, "wb") as f:
            pickle.dump(partitions[partition_key], f)
        counter += 1


def prebatch(data_path, config):
    prebatcher.prebatch(data_path, config)


def vocab(data_path, config):
    print("Generating vocab file %s from from files in %s" % (config.vocab_file, config.list_file))
    list_file = os.path.join(data_path, config.list_file)
    data_raw, _, _ = pyreader.read_data(data_path, list_file, gen_def_positions=False)
    data_raw = [[t[0] for t in file_tokens] for file_tokens in data_raw]

    force_include = astwalker.possible_identifiers() if config.vocab_incl_ids else None
    word_to_id = pyreader.build_vocab(data_raw, config.oov_threshold, force_include=force_include)
    outpath = os.path.join(data_path, config.vocab_file)
    with open(outpath, "wb") as f:
        pickle.dump(word_to_id, f)

    print("Vocab size: %d" % len(word_to_id))
    print("Generated file %s" % outpath)


mode_flags = {
    "train": (FLAGS.train, train),
    "test": (FLAGS.test, test2),
    "preprocess": (FLAGS.preprocess, preprocess),
    "prebatch": (FLAGS.prebatch, prebatch),
    "vocab": (FLAGS.vocab, vocab)
}

flag_validations = {
    "lambda_type": (lambda flags: any(f in flags.lambda_type for f in ["fixed", "state", "att", "input"]),
                    "lambda_type must be one of 'fixed', 'state' or 'att' or 'input'"),
    "attention": (lambda flags: flags.attention is None or all([f in ["full", "identifiers"] for f in flags.attention]),
                  "attention can only contain 'full' or 'identifiers', join using +"),
    "attention_variant": (lambda flags: flags.attention_variant is None or flags.attention_variant
                            in ["input", "output", "exlambda", "keyvalue", "baseline"],
                          "attention_variant must be 'input' or 'output' or 'exlambda' or 'experimental'"),
    "attention+variant": (lambda flags: not flags.attention or flags.attention and flags.attention_variant is not None,
                          "attention_variant parameter must be specified when using attention")
}


def validate_flags():
    for validation in flag_validations.values():
        if not validation[0](FLAGS):
            print("Configuration error: %s" % validation[1])
            sys.exit()


def print_flags(flags):
    for flag in flags.__flags:
        val = getattr(flags, flag)
        if not isinstance(val, bool) or val:
            print("%s=%s" % (flag, val))
    print()
    print()


def adjust_flags():
    if FLAGS.attention:
        FLAGS.attention = FLAGS.attention.split("+")
        if "identifiers" in FLAGS.attention:
            FLAGS.attention.extend(["identifiers"] * (len(astwalker.possible_types()) - 1))


if __name__ == "__main__":
    adjust_flags()
    validate_flags()
    print_flags(FLAGS)

    for flag in mode_flags:
        if mode_flags[flag][0]:
            print("Tensorflow version: %s" % tf.__version__)
            print("Running %s\n-------------\n" % flag)
            t0 = datetime.datetime.now()
            mode_flags[flag][1](FLAGS.data_path, FLAGS)
            t1 = datetime.datetime.now()
            print("Completed %s, total time taken: %s" % (flag, t1 - t0))

    if all(not flag[0] for flag in mode_flags.values()):
        print("No valid run mode argument specified, not doing anything")
        print("You must specify one of %s " % ", ".join(mode_flags.keys()))
