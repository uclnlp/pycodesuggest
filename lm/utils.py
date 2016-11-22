import os
import sys
import shutil
import copy
from glob import iglob
import pickle
import itertools
import datetime
import numpy as np

from model import *


def get_file_list(config, data_path, pattern, description):
    files = [y for x in os.walk(data_path) for y in iglob(os.path.join(x[0], pattern))]

    if len(files) == 0:
        print("No partitions found for %s data, exiting..." % description)
        sys.exit()

    print("Found %d%s partitions for %s data"
          % (len(files), " prebatched" if config.use_prebatched else "", description))
    if config.num_partitions:
        print("But only using %d due to num_partitions parameter" % config.num_partitions)
    files = files[:config.num_partitions]
    return files


def copy_temp_files(files, temp_dir):
    temp_files = []
    for file in files:
        target_file = os.path.split(file)[1]
        target_file = os.path.join(temp_dir, target_file)
        shutil.copy2(file, target_file)
        temp_files.append(target_file)
    return temp_files


def create_model(config, is_training):
    if config.attention and config.attention_variant == "input":
        return AttentionModel(is_training=is_training, config=config)
    elif config.attention and config.attention_variant == "output":
        return AttentionOverOutputModel(is_training=is_training, config=config)
    elif config.attention and config.attention_variant == "keyvalue":
        return AttentionKeyValueModel(is_training=is_training, config=config)
    elif config.attention and config.attention_variant == "exlambda":
        return AttentionWithoutLambdaModel(is_training=is_training, config=config)
    elif config.attention and config.attention_variant == "baseline":
        return AttentionBaselineModel(is_training=is_training, config=config)
    else:
        return BasicModel(is_training=is_training, config=config)


def attention_masks(attns, masks, length):
    lst = []
    if "full" in attns:
        lst.append(np.ones([1, length]))
    if "identifiers" in attns:
        lst.append(masks[:, 0:length] if len(masks.shape) == 2 else np.reshape(masks[0:length], [1, length]))

    return np.transpose(np.concatenate(lst)) if lst else np.zeros([0, length])


class FlagWrapper:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

    def __getattr__(self, name):
        return self.__dict__['__flags'][name]


def copy_flags(flags):
    dict_copy = copy.copy(flags.__dict__)
    return FlagWrapper(dict_copy)


def identity_map(x):
    return x


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


# from
#   http://stackoverflow.com/questions/33759623/tensorflow-how-to-restore-a-previously-saved-model-python
def save_model(saver, sess, path, model, config):
    if not os.path.exists(path):
        os.makedirs(path)

    now = datetime.now().strftime("%Y-%m-%d--%H-%M--%f")
    out_path = os.path.join(path, now + "/")

    tf.train.write_graph(model.graph.as_graph_def(), out_path, 'model.pb', as_text=False)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    saver.save(sess, os.path.join(out_path, "model.tf"))

    latest_path = os.path.join(path, "latest")
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(now, latest_path)
    return out_path


def load_model(sess, path):
    load_variables(sess, os.path.join(path, "model.tf"), tf.trainable_variables())


def load_variables(session, path, variables):
    saver = tf.train.Saver(variables)
    saver.restore(session, path)

