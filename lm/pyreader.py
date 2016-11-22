import tokenize
import os
import collections
from io import StringIO

import numpy as np
import sys
import itertools
import math

import astwalker
from normalisation import get_source_tree

pad_token, pad_id = "§PAD§", 0
oov_token, oov_id = "§OOV§", 1
indent_token = "§<indent>§"
dedent_token = "§<dedent>§"
number_token = "§NUM§"


class Container:
    def __init__(self, inputs, targets, actual_lengths, masks, identifier_usage):
        self.inputs = inputs
        self.targets = targets
        self.actual_lengths = actual_lengths
        self.num_sequences = len(inputs)
        self.masks = masks
        self.identifier_usage = identifier_usage


# Reads and tokensises all python files in the given path
def read_data(path, listfile, word_to_id=None, gen_def_positions=True):
    if isinstance(listfile, list):
        python_files = [os.path.join(path, f) for f in listfile]
    else:
        with open(listfile) as f:
            python_files = [os.path.join(path, x) for x in f.read().splitlines()]

    mapping = (lambda x: x) if word_to_id is None else (lambda x: word_to_id.get(x, oov_id))

    data = []
    definition_positions = []
    identifier_usage = []
    for filename in python_files:
        try:
            source, tree = get_source_tree(filename)
            tokens = tokenize.generate_tokens(StringIO(source).readline)

            data.append([(mapping(preprocess(tokenType, tokenVal)), start) for tokenType, tokenVal, start, _, _
                         in tokens
                         if tokenType != tokenize.COMMENT and
                         not tokenVal.startswith("'''") and
                         not tokenVal.startswith('"""') and
                         (tokenType == tokenize.DEDENT or tokenVal != "")])

            if gen_def_positions:
                walker = astwalker.ASTWalker()
                walker.walk(tree)
                definition_positions.append(walker.definition_positions)
                identifier_usage.append(walker.name_usage)

        except:
            print("Error when tokenizing %s: %s" % (filename, sys.exc_info()[0]))

    return data, definition_positions, identifier_usage


def preprocess(tokentype, tokenval):
    if tokentype == tokenize.NUMBER:
        return number_token

    elif tokentype == tokenize.INDENT:
        return indent_token

    elif tokentype == tokenize.DEDENT:
        return dedent_token

    return tokenval


def build_vocab(data, oov_threshold, force_include=None):
    force_include = force_include or []
    counter = collections.Counter(itertools.chain(itertools.chain(*data), force_include))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    count_pairs = (p for p in count_pairs if p[1] > oov_threshold or p[0] in force_include)

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(2, len(words) + 2)))
    word_to_id[oov_token] = oov_id
    word_to_id[pad_token] = pad_id
    return word_to_id


def get_data(path, listfile, seq_length, word_to_id):
    all_data, def_positions, identifier_usage = read_data(path, listfile, word_to_id)
    identifier_types = astwalker.possible_types()
    num_masks = len(identifier_types)
    def_positions = [[[t[1] for t in fp if t[0] == k] for k in identifier_types] for fp in def_positions]
    # def_positions = [[t[1] for t in file_positions] for file_positions in def_positions]
    num_masks = len(identifier_types)
    #num_masks = 1
    data = []

    # Prevent indent and dedent tokens from being flagged as variables, which can occur because dedent
    # in particular takes up no columns
    non_vars = [word_to_id[indent_token], word_to_id[dedent_token]]

    for j in range(len(all_data)):
        filedata = list(all_data[j])
        file_positions = def_positions[j]
        identifier_positions = identifier_usage[j]

        num_sequences = math.ceil(len(filedata) / seq_length)
        input_data = np.zeros([num_sequences, seq_length]).astype("int64")
        targets = np.zeros([num_sequences, seq_length]).astype("int64")
        masks_data = np.zeros([num_sequences, num_masks, seq_length]).astype("bool")
        id_usage_data = np.zeros([num_sequences, seq_length]).astype("bool")
        actual_lengths = []

        for i in range(num_sequences):
            x = [t[0] for t in filedata[i * seq_length:(i + 1) * seq_length]]
            y = [t[0] for t in filedata[i * seq_length + 1:(i + 1) * seq_length + 1]]
            masks = [[t[1] in fp and t[0] not in non_vars for fp in file_positions]
                     for t in filedata[i * seq_length:(i + 1) * seq_length]]
            # masks = [t[1] in file_positions and t[0] not in non_vars
            #          for t in filedata[i * seq_length:(i + 1) * seq_length]]
            ids = [t[1] in identifier_positions for t in filedata[i * seq_length + 1:(i + 1) * seq_length + 1]]

            actual_length_x = len(x)
            actual_length_y = len(y)

            input_data[i, :actual_length_x] = x
            targets[i, :actual_length_y] = y
            masks_data[i, :, :actual_length_x] = np.transpose(masks)
            id_usage_data[i, :actual_length_y] = ids

            actual_lengths.append(actual_length_y)

        container = Container(input_data, targets, actual_lengths, masks_data, id_usage_data)
        data.append(container)

    return data


def partition_data(data, num_partitions):
    total_sequences = sum(c.num_sequences for c in data)
    bucket_size = total_sequences // num_partitions
    data = sorted(data, key=lambda c: c.num_sequences)
    buckets = {}
    counter = 0
    for i in range(num_partitions):
        bucket = []
        assigned_sequences = 0
        while assigned_sequences < bucket_size and counter < len(data):
            bucket.append(data[counter])
            assigned_sequences += data[counter].num_sequences
            counter += 1
        buckets[i] = bucket

    return buckets


if __name__ == "__main__":
    path = "/Users/avishkar/pyRepos"
    listfile = "/Users/avishkar/pyRepos/train_files.txt"
    raw_data, _, _ = read_data(path, listfile)
    data_raw = [[t[0] for t in file_tokens] for file_tokens in raw_data]
    word_to_id = build_vocab(data_raw, 0)
    inv_map = {v: k for k, v in word_to_id.items()}
    data = get_data(path, listfile, 20, word_to_id)

    inputs = [inv_map[int(i)] for i in np.nditer(data[0].inputs)]
    masks = [bool(i) for i in np.nditer(data[0].masks)]
    output = []
    for i, input in enumerate(inputs):
        output.append("§§" + input + "§§" if masks[i] else input.replace("§<indent>§", "\t").replace("§<dedent>§", ""))

    print("Identifier introduced:")
    print(" ".join(output))

    targets = [inv_map[int(i)] for i in np.nditer(data[0].targets)]
    id_usages = [bool(i) for i in np.nditer(data[0].identifier_usage)]
    output2 = []
    for i, target in enumerate(targets):
        output2.append("§§" + target + "§§" if id_usages[i] else target.replace("§<indent>§", "\t").replace("§<dedent>§", ""))

    print("Identifier Used:")
    print(" ".join(output2))
