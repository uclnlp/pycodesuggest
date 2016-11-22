import os
import pickle
from glob import iglob

from batcher import QueuedSequenceBatcher


def prebatch(data_path, config):
    print("Running pre-batch processing")
    run_prebatch(data_path, "train", config)
    run_prebatch(data_path, "valid", config)


def run_prebatch(data_path, set, config):
    pattern = 'all_' + set + '_data.dat.part*'
    files = [y for x in os.walk(data_path) for y in iglob(os.path.join(x[0], pattern))]
    print("Found %d partitions for %s data" % (len(files), set))
    for file in files:
        batcher = QueuedSequenceBatcher([file], config.input_seq_length, config.batch_size, description=set,
                                        attns=config.attention)
        file_data = []
        for batch in batcher:
            batch_acc = []
            for sequence in batcher.sequence_iterator(batch):
                batch_acc.append(sequence)
            file_data.append(batch_acc)

        target_file = os.path.join(data_path, 'prebatched_' + set + '_data.part' + file[-1])
        with open(target_file, 'wb') as f:
            pickle.dump(file_data, f)
        print("Wrote pre-batched partition %s" % target_file)

