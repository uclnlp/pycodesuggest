import re
import ast
import pickle
import sys
import getopt
import astor
import os
import astwalker
from glob import iglob


def normalise(path):

    split = os.path.split(path)
    base = split[0]
    dirname = split[1]

    normalised_target_path = os.path.join(base, dirname + "_normalised")
    processed_file_path = os.path.join(path, "processed.txt")

    print("Writing normalised files to %s" % normalised_target_path)

    python_files = [y[len(path)+1:] for x in os.walk(path) for y in iglob(os.path.join(x[0], '*.py'))]

    # For debugging
    # python_files = ["debug/test.py"]
    # python_files = ["web2py/gluon/contrib/memcache/memcache.py"]

    processed_files = []
    initial_processed = 0
    syntax_errors = []
    filenotfound_errors = []
    errors = []
    skipped = []

    if os.path.exists(processed_file_path):
        print("Found processed files from previous session, continuing...")
        with open(processed_file_path) as p:
            processed_files = p.read().splitlines()
            initial_processed = len(processed_files)

    def complete():
        write_output(processed_file_path, processed_files)
        print("Processed files: %d\nSyntax errors: %d\nFile not found errors: %d\nOther errors: %d\nSkipped: %d" %
              (len(processed_files) - initial_processed, len(syntax_errors),
               len(filenotfound_errors), len(errors), len(skipped)))

    for filename in python_files:
        if filename in processed_files:
            skipped.append(filename)
            continue

        error = False
        try:
            input_file = os.path.join(path, filename)
            normalised_target_file = os.path.join(normalised_target_path, filename)
            source, tree = get_source_tree(input_file)
        except SyntaxError:
            syntax_errors.append(filename)
            continue
        except FileNotFoundError:
            filenotfound_errors.append(filename)
            continue
        except KeyboardInterrupt:
            print("Keyboard interrupt, saving...")
            complete()
            sys.exit()
        except:
            print("Failed to parse %s due to %s" % (filename, sys.exc_info()[0]))
            errors.append((filename, sys.exc_info()[0]))
            continue

        # AST variable replacement and formatting
        try:
            walker = astwalker.ASTWalker()
            # walker.randomise = False  # For debugging
            walker.walk(tree)
            walker.process_replace_queue()
            ast_source = astor.to_source(tree)
            writefile(normalised_target_file, ast_source)
        except KeyboardInterrupt:
            print("Keyboard interrupt, saving...")
            complete()
            sys.exit()
        except:
            print("Failed to process normalisation for file %s" % filename)
            print(sys.exc_info()[0])
            error = True
            if len(python_files) == 1:
                raise

        if not error:
            processed_files.append(filename)

    complete()


def write_output(processed_file_path, processed_files):
    with open(processed_file_path, "w") as f:
        f.writelines([p + "\n" for p in processed_files])


def writemapping(mapping, map_file):
    os.makedirs(os.path.dirname(map_file), exist_ok=True)
    with open(map_file, 'wb') as f:
        pickle.dump(mapping, f)


def writefile(target, source):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w') as f:
        f.write(source)


def replace_source(source, name_dict):
    replace_order = ["Class", "function", "attribute", "arg", "var"]
    for scope in name_dict:
        for typename in set(replace_order).intersection(name_dict[scope].keys()):
            for name in name_dict[scope][typename]:
                if name is not None and name_dict[scope][typename][name] is not None:
                    source = re.sub(r"\b" + name + r"\b", name_dict[scope][typename][name], source)

    return source


def get_source_tree(filename):
    with open(filename, 'r') as f:
        fstr = f.read()
    fstr = fstr.replace('\r\n', '\n').replace('\r', '\n')
    if not fstr.endswith('\n'):
        fstr += '\n'
    return fstr, ast.parse(fstr, filename=filename)


def printusage():
    print('Usage: normalisation.py -p <data path>')


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

    normalise(path)

if __name__ == "__main__":
    main(sys.argv[1:])



