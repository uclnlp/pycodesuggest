import ast
import os
import tokenize
from glob import iglob
import hashlib
import sys
import getopt

from io import StringIO

import numpy as np

import collections

import itertools

import pickle

import astwalker
from normalisation import get_source_tree

train_proportion = 0.5
valid_proportion = 0.2

def split_files(path):
    subdirectories = [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

    python_files = [(directory, [y for x in os.walk(directory) for y in iglob(os.path.join(x[0], '*.py'))])
                for directory in subdirectories]

    train_files = []
    valid_files = []
    test_files = []

    train_split = int(len(python_files) * train_proportion)
    valid_split = train_split + int(len(python_files) * valid_proportion)
    for project in python_files[:train_split]:
        train_files.extend([f[len(path)+1:] for f in project[1]])

    for project in python_files[train_split:valid_split]:
        valid_files.extend([f[len(path)+1:] for f in project[1]])

    for project in python_files[valid_split:]:
        test_files.extend([f[len(path)+1:] for f in project[1]])

    def write_to_file(fname, lst):
        with open(os.path.join(path, fname), "w") as write_file:
            for f in lst:
                print(f, file=write_file)

    write_to_file("train_files.txt", train_files)
    write_to_file("valid_files.txt", valid_files)
    write_to_file("test_files.txt", test_files)


def corpus_stats(path):
    def stats(description, base_path, list_file):
        line_count = 0
        with open(os.path.join(base_path, list_file)) as l:
            files = l.readlines()
            file_count = len(files)
            for file in files:
                try:
                    f = open(os.path.join(base_path, file.strip()))
                    line_count += len(f.readlines())
                    f.close()
                except:
                    print("Skipping file %s due to %s" % (file, sys.exc_info()[0]))

        print("Statistics for %s" % description)
        print("File count: %d" % file_count)
        print("Line count: %d" % line_count)
        print("---------------------------")

    stats("Train", path, "train_files.txt")
    stats("Valid", path, "valid_files.txt")
    stats("Test", path, "test_files.txt")


def check_duplicates(path):

    subdirectories = [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

    python_files = [(directory, [y for x in os.walk(directory) for y in iglob(os.path.join(x[0], '*.py'))])
                    for directory in subdirectories]

    def hashfile(path, blocksize=65536):
        afile = open(path, 'rb')
        hasher = hashlib.md5()
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
        afile.close()
        return hasher.hexdigest()

    dups = {}
    for filename in python_files:
        file_hash = hashfile(filename)
        if file_hash in dups:
            dups[file_hash].append(path)
        else:
            dups[file_hash] = [path]

    results = list(filter(lambda x: len(x) > 1, dups.values()))
    if len(results) > 0:
        print('Duplicate candidates Found:')
        for result in results:
            for subresult in result:
                print('\t\t%s' % subresult)
            print('___________________')

    else:
        print('No duplicate files found.')


def count_variables(path):

    subdirectories = [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

    python_files = [(directory, [y for x in os.walk(directory) for y in iglob(os.path.join(x[0], '*.py'))])
                    for directory in subdirectories]

    max_count = {}
    max_file = {}
    syntax_error_count = 0

    for pair in python_files:
        for file in pair[1]:
            try:
                source, tree = get_source_tree(file)
                walker = astwalker.ASTWalker()
                walker.walk(tree)
                counters = {}
                for scope in walker.names:
                    for typename in walker.names[scope]:
                        counters[typename] = counters.get(typename, 0) + len(walker.names[scope][typename])

                for typename in counters:
                    if counters[typename] > max_count.get(typename, 0):
                        max_count[typename] = counters[typename]
                        max_file[typename] = file
            except SyntaxError:
                syntax_error_count += 1
            except:
                print("Skipping file %s due to %s" % (file, sys.exc_info()[0]))

    print("%d syntax errors" % syntax_error_count)
    print(max_count)
    print(max_file)


def read_data(path, listfile):

    with open(listfile) as f:
        python_files = [os.path.join(path, x) for x in f.read().splitlines()]

    data = []

    for filename in python_files:
        try:
            with open(filename, 'r') as f:
                tokens = tokenize.generate_tokens(f.readline)

                data.append([tokenVal for tokenType, tokenVal, start, _, _
                             in tokens
                             if tokenType != tokenize.COMMENT and
                             not tokenVal.startswith("'''") and
                             not tokenVal.startswith('"""') and
                             tokenVal != "" and
                             tokenType != tokenize.INDENT])
        except:
            print("Error when tokenizing %s: %s" % (filename, sys.exc_info()[0]))

    return data


def token_distribution(path):
    data = read_data(path, os.path.join(path, "test_files.txt"))
    counter = collections.Counter(itertools.chain(*data))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    tokens = 0
    vocab = 0
    for k,g in itertools.groupby(count_pairs, lambda p: p[1]):
        tokens += k
        vocab_size = len(list(g))
        vocab += vocab_size
        print("%d, %d" % (k, vocab_size))

    print("%d : %d" % (tokens, vocab))


def count_identifiers():
    data_path = "/Volumes/SAMSUNG/pythonRepos_vars_format"
    map_path = "/Volumes/SAMSUNG/pythonRepos_map"
    with open(os.path.join(data_path, "train_files.txt")) as f:
        python_files = [x for x in f.read().splitlines()]

    identifiers = []
    for file in python_files:
        map_file = os.path.join(map_path, file + ".map")
        with open(map_file, 'rb') as mf:
            mapping = pickle.load(mf)

        for k_outer in mapping:
            for type in mapping[k_outer]:
                for identifier in mapping[k_outer][type]:
                    identifiers.append(identifier)

    print("Total identifiers: %d " % len(identifiers))
    print("Unique identifiers: %d" % len(set(identifiers)))


def id_distance(path):
    types = ["var", "function", "Class", "attribute", "arg"]

    def type(target_name):
        for t in types:
            if target_name.startswith(t):
                return t
        return None

    def process_queue(walker):
        position_maps = {}
        for scope, typenames, candidate, class_scope in walker.queue:
            name = candidate.id if isinstance(candidate, ast.Name) else candidate.attr
            currentPosition = (candidate.lineno, candidate.col_offset)
            target_name = walker.lookup_name(scope, typenames, name)
            if target_name is None and class_scope:
                for s in walker.linked_class_scopes.get(scope,[]):
                    target_name = walker.lookup_name(s, typenames, name)

            if target_name is not None:
                def_position = walker.name_mapping[target_name]
                vartype = type(target_name)
                if def_position not in position_maps:
                    position_maps[(def_position, vartype)] = []
                position_maps[(def_position, vartype)].append(currentPosition)

        return position_maps

    def calc_distances(source, targets, tokens):
        if source not in tokens:
            return []

        source_i = tokens.index(source)

        def dist(t):
            if t not in tokens:
                return -1
            t_i = tokens.index(t)
            return abs(t_i - source_i)

        return [x for x in [dist(t) for t in targets] if x >= 0]

    type_distances = {}
    for t in types:
        type_distances[t] = []

    with open(os.path.join(path, "valid_files.txt")) as l:
        files = l.readlines()
        for i, file in enumerate(files):
            if i % 100 == 0: print(file)
            source, tree = get_source_tree(os.path.join(path, file.strip()))
            tokens = tokenize.generate_tokens(StringIO(source).readline)
            tokens = [start for _, _, start, _, _ in tokens]
            walker = astwalker.ASTWalker()
            walker.walk(tree)
            pos_maps = process_queue(walker)

            for k in pos_maps:
                type_distances[k[1]].extend(calc_distances(k[0], pos_maps[k], tokens))

    for t in type_distances:
        data = np.array(type_distances[t])
        print(t)
        print("Minimum: %d" % np.min(data))
        print("Q1: %d" % np.percentile(data, 25))
        print("Median: %d" % np.percentile(data, 50))
        print("Q3: %d" % np.percentile(data, 75))
        print("90th Percentile: %d" % np.percentile(data, 90))
        print("Maximum: %d" % np.max(data))
        print()


def printusage():
    print('Usage: processFiles.py -p <repo path>')


def main(argv):
    path = ''
    try:
        opts, args = getopt.getopt(argv, "hp:", ["path="])
    except getopt.GetoptError:
        printusage()
        raise

    for opt, arg in opts:
        if opt == '-h':
            printusage()
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg

    if path == '':
        printusage()
        sys.exit(2)

    split_files(path)
    corpus_stats(path)


if __name__ == "__main__":
    main(sys.argv[1:])




