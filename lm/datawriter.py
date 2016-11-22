import tensorflow as tf
import os


def write_data(data, path):
    writer = tf.python_io.TFRecordWriter(path)
    all_inputs = data[0]
    all_targets = data[1]

    for i in range(len(all_inputs)):
        input = all_inputs[i].tolist()
        target = all_targets[i].tolist()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'input': tf.train.Feature(int64_list=tf.train.Int64List(value=input)),
                    'target': tf.train.Feature(int64_list=tf.train.Int64List(value=target))
                }
            )
        )

        serialised = example.SerializeToString()
        writer.write(serialised)


def read_and_decode_single_example(filename_queue, seq_length):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feats = {
        'input': tf.FixedLenFeature([seq_length], tf.int64, [-1]*seq_length),
        'target': tf.FixedLenFeature([seq_length], tf.int64, [-1]*seq_length)
    }

    features = tf.parse_single_example(serialized_example, feats)

    return features['input'], features['target']


def inputs(filename, batch_size, num_epochs, seq_length):
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, shuffle=True)

    input, targets = read_and_decode_single_example(filename_queue, seq_length)
    #input, targets = tf.train.batch([input, targets], batch_size, num_threads=2)
    input, targets = tf.train.batch([input, targets], batch_size)
    return input, targets

