import getopt
import os
import sys
import reader


def train_ngram(data_path, list_file, out_file):
    list_path = os.path.join(data_path, list_file)

    with open(out_file, "w") as f:
        for file_data in reader.read_data(data_path, list_path):
            f.write(" ".join(file_data) + "\n")


def printusage():
    print("Usage: ngram -p <Data path> -l <list_file> -o <out_file>")


def main(argv):
    path = ''
    list_file = ''
    out_file = ''
    try:
        opts, args = getopt.getopt(argv, "hp:l:o:", ["path=", "list=", "outfile="])
    except getopt.GetoptError:
        printusage()
        raise

    for opt, arg in opts:
        if opt == '-h':
            printusage()
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-l", "--list"):
            list_file = arg
        elif opt in ('-o', "--outfile"):
            out_file = arg

    if path == '' or list_file == '' or out_file == '':
        printusage()
        sys.exit(2)

    train_ngram(path, list_file, out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
