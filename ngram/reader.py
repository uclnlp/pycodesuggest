# Reads and tokensises all python files in the given path
import os
import tokenize

import itertools

pad_token, pad_id = "§PAD§", 0
oov_token, oov_id = "§OOV§", 1
number_token = "§NUM§"


def read_data(path, listfile):
    if isinstance(listfile, list):
        python_files = [os.path.join(path, f) for f in listfile]
    else:
        with open(listfile) as f:
            python_files = [os.path.join(path, x) for x in f.read().splitlines()]

    for filename in python_files:
        try:
            with open(filename) as f:
                tokens = list(tokenize.generate_tokens(f.readline))

                yield [preprocess(tokenType, tokenVal) for tokenType, tokenVal, _, _, _
                       in tokens
                       if tokenType != tokenize.COMMENT and
                       not tokenVal.startswith("'''") and
                       not tokenVal.startswith('"""') and
                       (tokenType == tokenize.DEDENT or tokenVal != "")]
        except:
            pass


def preprocess(tokentype, tokenval):
    if tokentype == tokenize.NUMBER:
        return number_token

    elif tokentype == tokenize.INDENT:
        return "<indent>"

    elif tokentype == tokenize.DEDENT:
        return "<dedent>"

    # Need to replace spaces with some other character because the ngram processor
    # splits on spaces
    return tokenval.replace(" ", "§").replace("\n", "<newline>")
