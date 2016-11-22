import pickle
import matplotlib.pyplot as plt

import tempfile
from matplotlib.colors import NoNorm

import astwalker
import pyreader
from BeamSearchTree import BeamSearchTreeNode
from Trainer import get_initial_state, construct_feed_dict, extract_results, get_evals
from beamSearch import find_path
from hooks import GeneratorHook, PerplexityHook, TopKAccuracyHook
from pyreader import oov_id
from batcher import PreBatched, QueuedSequenceBatcher
from termcolor import print_color, gray, rgb
from tfrnn.util import load_model
from utils import *
import seaborn as sns


def eval(data_path, config):
    if not config.model_path:
        raise ValueError("model_path parameter is required when testing")

    with open(os.path.join(data_path, "mapping.map"), "rb") as dict_file:
        word_to_id = pickle.load(dict_file)

    with open(os.path.join(config.model_path, "config.pkl"), "rb") as config_file:
        model_config_dict = pickle.load(config_file)
        model_config_dict["batch_size"] = config.batch_size
        if "attention" not in model_config_dict:
            model_config_dict["attention"] = config.attention
        model_config = FlagWrapper(model_config_dict)

    config.vocab_size = len(word_to_id)
    print("Vocab size: %d" % config.vocab_size)
    run_tests(config, model_config, data_path, word_to_id)


def run_tests(config, model_config, data_path, word_to_id):
    with tf.Graph().as_default(), tf.Session() as session:
        generator_config = copy_flags(model_config)
        generator_config.seq_length = 1
        generator_config.batch_size = 1

        with tf.variable_scope("model", reuse=None):
            model = create_model(model_config, False)
        with tf.variable_scope("model", reuse=True):
            generator_model = create_model(generator_config, False)

        init = tf.initialize_all_variables()
        session.run(init)
        load_model(session, config.model_path)

        print("Sample generation:")
        # generator = GeneratorHook(generator_model, word_to_id, model_config.attention, sample_length=50)
        # generator(session, 1, 0, generator_model.logits, 0, 0)

        # top_predict = TopPredictions(generator_model, word_to_id, model_config.attention)
        # top_predict(session)

        beam_search = BeamSearch2(generator_model, word_to_id, model_config.attention, 5)
        beam_search(session, 2)

        test_pattern = config.data_pattern.replace("{-type-}", "test") + ".part*"
        files = get_file_list(config, data_path, test_pattern, "test")


        if config.copy_temp:
            temp_dir = tempfile.mkdtemp()
            print("Copying data files to %s" % temp_dir)
            files = copy_temp_files(files, temp_dir)

        #batcher = PreBatched(files, config.batch_size, description="test") if config.use_prebatched \
        #    else QueuedSequenceBatcher(files, config.seq_length, config.batch_size, description="test",
        #                               attns=model_config.attention)


        #perplexity = PerplexityHook(None, model, batcher)
        #perplexity(session, 1, 0, model.logits, 0, 0)

        #accuracy = TopKAccuracyHook(None, model, batcher, [1, 5])
        #accuracy(session, 1, 0, model.logits, 0, 0)

        # list = ListPredictions(generator_model, word_to_id, data_path)
        # list(session)

        #if model.is_attention_model:
            # vis = AttentionVisualiser(generator_model, word_to_id, data_path,
            #                          model_config.max_attention, model_config.attention)
            # vis(session)

            #vis = LaggedAttentionVisualisation(generator_model, word_to_id, data_path,
            #                                   model_config.max_attention, model_config.attention)

            #vis(session)

        # profiler = Profiler(model, batcher)
        # profiler.profile(session)

        if config.copy_temp:
            shutil.rmtree(temp_dir)


class Profiler():
    '''Runs a single step of training and outputs profile trace information'''

    def __init__(self, model, batcher):
        self.model = model
        self.batcher = batcher

    def profile(self, session):
        evals = [self.model.cost]
        for batch in self.batcher:
            state, att_states, att_ids, att_counts = get_initial_state(self.model)

            for seq_batch in self.batcher.sequence_iterator(batch):
                feed_dict = construct_feed_dict(self.model, seq_batch, state, att_states, att_ids, att_counts)
                run_metadata = tf.RunMetadata()
                session.run(evals, feed_dict=feed_dict,
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata)
                break

        from tensorflow.python.client import timeline
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open('timeline.ctf.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format())


class AttentionVisualiser:
    def __init__(self, model, map, data_path, max_attention, attns):
        if model.seq_length != 1 or model.batch_size != 1:
            raise ValueError("Attention Visualiser only works with a model with a batch size and sequence length of 1")

        self.model = model
        self.map = map
        self.inverse_map = {v: k for k, v in map.items()}
        self.data_path = data_path
        self.max_attention = max_attention
        self.attns = attns

        '''
        self.test_cases = [
            ["class", "MyClass", ":", "\n", "    ", "def", "__init__", "(",
             "self", ",", "model", ")", ":", "\n", "        ", "self", "."]
        ]
        self.var_masks = [
            [0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0]
        ]
        '''
        self.test_cases = []
        self.var_masks = []
        self.test_cases = [[self.map.get(t, oov_id) for t in testcase] for testcase in self.test_cases]
        # self.test_files = ['mher/flower/pavement.py']
        self.test_files = ['debug/test_classes.py']

    def __call__(self, session):
        test_file_containers = []
        if self.test_files:
            test_file_containers = pyreader.get_data(self.data_path, self.test_files, 1, self.map)

        data = list(zip(self.test_cases, self.var_masks)) + \
               list(zip((list(flatmap(identity_map, c.inputs)) for c in test_file_containers),
                        (list(flatmap(identity_map, c.masks)) for c in test_file_containers)))

        for testcase, var_mask in data:
            if len(testcase) != len(var_mask):
                raise ValueError("Length of testcase does not match corresponding variable mask: %s" % testcase)

            print("----------Test case----------\n")
            evals = get_evals([self.model.predict], self.model)
            state, att_states, att_ids, att_counts = get_initial_state(self.model)

            prev_mask = False
            attns = []
            plot_data = np.zeros([len(testcase), self.max_attention])
            lambda_data = np.zeros([len(testcase), 2])
            annotations = np.empty([len(testcase), self.max_attention], dtype=object)
            y_labels = []
            predicted_token = ""

            for i, (token, mask) in enumerate(zip(testcase, var_mask)):
                att_mask = attention_masks(self.attns, np.array([mask]), 1)
                data = (np.array([[token]]), np.array([[1]]), np.array([att_mask]), np.array([[1]]), np.array([1]))
                feed_dict = construct_feed_dict(self.model, data, state, att_states, att_ids, att_counts)
                results = session.run(evals, feed_dict=feed_dict)
                prediction, state, att_states, att_ids, alpha_states, att_counts, lambda_vec = \
                    extract_results(results, evals, 1, self.model)
                predicted = np.argmax(prediction)

                if prev_mask:
                    if len(attns) >= self.max_attention:
                        attns = attns[1:]
                    attns.append(prev_token)

                prev_mask = mask
                prev_token = self.inverse_map[token]

                plot_data[i, :] = alpha_states[0][0] * (lambda_vec[0, 1] if lambda_vec[0, 1] < 0.1 else 1)
                lambda_data[i, :] = lambda_vec
                labels = [""] * (self.max_attention-len(attns)) + attns
                annotations[i, :] = labels

                current_token = self.inverse_map[token]
                current_token = "%s%s%s" % ("** " if current_token == predicted_token else "", "(*)" if mask else "", current_token)
                y_labels.append(current_token)
                predicted_token = self.inverse_map[predicted]

            fig, (ax_data, ax_lambda) = plt.subplots(1, 2, gridspec_kw={
                'width_ratios': [self.max_attention, 2]
            })

            blank_x_labels = [""] * self.max_attention
            blank_y_labels = [""] * len(testcase)
            lambda_x_labels = ["LM", "Att"]
            sns.set(font_scale=1.2)
            sns.set_style({"savefig.dpi": 100})
            plt.yticks(rotation=0)
            ax_data = sns.heatmap(plot_data, ax=ax_data, cmap=plt.cm.Blues, linewidths=.1, annot=annotations,
                                  fmt="", vmin=0, vmax=1, cbar=False, xticklabels=blank_x_labels, yticklabels=y_labels,
                                  annot_kws={"size": 9})
            ax_lambda = sns.heatmap(lambda_data, ax=ax_lambda, cmap=plt.cm.Blues, linewidths=.1, annot=False,
                                    fmt="", vmin=0, vmax=1, cbar=False, xticklabels=lambda_x_labels, yticklabels=blank_y_labels,
                                    annot_kws={"size": 9})

            ax_data.set_yticklabels(ax_data.yaxis.get_majorticklabels(), rotation=0)
            ax_lambda.xaxis.tick_top()
            fig.set_size_inches(int(self.max_attention)*1.3, int(len(plot_data)/3))

            fig.savefig('./out/attention2.png')
            print("Generated file attention2.png")


def plot_heatmap(ax, data, x_labels, y_labels, rotate=0):
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, norm=NoNorm())

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

    ax.xaxis.tick_top()
    ax.set_xticklabels(x_labels, minor=False, rotation=rotate)
    ax.set_yticklabels(y_labels, minor=False)

token_mapping = {
    "§<indent>§": "<indent>",
    "§<dedent>§": "<dedent>",
    "\n": "<newline>"
}


def clean_token(token):
    return token_mapping.get(token, token)


class ListPredictions:
    def __init__(self, model, map, data_path):
        self.model = model
        self.map = map
        self.inverse_map = {v: k for k, v in map.items()}
        self.data_path = data_path
        self.test_files = ['debug/MyAttention.py']
        self.prediction_op = tf.argmax(model.predict, 1)

    def __call__(self, session):
        test_file_containers = pyreader.get_data(self.data_path, self.test_files, 1, self.map)
        data = list(zip((list(flatmap(identity_map, c.inputs)) for c in test_file_containers),
                        (list(flatmap(identity_map, c.masks)) for c in test_file_containers)))

        for testcase, var_mask in data:
            print("----------Test case----------\n")
            evals = get_evals([self.prediction_op], self.model)
            state, att_states, att_counts = get_initial_state(self.model)
            predicted_tokens = []

            for i, (token, mask) in enumerate(zip(testcase, var_mask)):
                data = (np.array([[token]]), np.array([[1]]), np.array([0]), np.array([1]))
                feed_dict = construct_feed_dict(self.model, data, state, att_states, att_counts)
                results = session.run(evals, feed_dict=feed_dict)
                prediction, state, att_states, att_counts, att_vec, lambda_vec = \
                    extract_results(results, evals, 1, self.model)
                predicted_token = self.inverse_map[prediction[0][0]].replace("\n", "<newline>")
                current_token = self.inverse_map[token].replace("\n", "<newline>")
                predicted_tokens.append(current_token + " ; " + predicted_token)

            print("\n".join(predicted_tokens))


class LaggedAttentionVisualisation:
    def __init__(self, model, map, data_path, max_attention, attns):
        self.model = model
        self.map = map
        self.inverse_map = {v: k for k, v in map.items()}
        self.data_path = data_path
        self.max_attention = max_attention
        self.test_files = ['debug/MyAttention.py']
        self.attns = attns
        self.max_display = 50

    def __call__(self, session):
        test_file_containers = pyreader.get_data(self.data_path, self.test_files, 1, self.map)
        data = list(zip((list(flatmap(identity_map, c.inputs)) for c in test_file_containers),
                        (list(flatmap(identity_map, c.masks)) for c in test_file_containers)))

        #tokens = ["def", "function234", "(", "arg289", ")", ":", "\n", "§<indent>§", "with", "open", "(", "§OOV§", ",", "'w'", ")", "as", "f|var76", ":", "\n", "§<indent>§", "f|var76", "."]
        #tokens = ["def", "function234", "(", "arg289", ")", ":", "\n", "§<indent>§", "with", "open", "(", "§OOV§", ",", "'r'", ")", "as", "f|var76", ":", "\n", "§<indent>§", "var91", "=", "f|var76", "."]
        #data = [([map_token(self.map, t) for t in tokens],
        #         [np.array([False]) for _ in tokens])]

        for testcase, var_mask in data:
            print("----------Test case----------\n")
            evals = get_evals([self.model.predict], self.model)
            state, att_states, att_counts = get_initial_state(self.model)

            accumulated_tokens = []
            plot_data = np.zeros([len(testcase), self.max_display])
            annotations = np.empty([len(testcase), self.max_display], dtype=object)
            y_labels = []
            predicted_labels = []

            for i, (token, mask) in enumerate(zip(testcase, var_mask)):
                att_mask = attention_masks(self.attns, np.array([mask]), 1)
                data = (np.array([[token]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
                feed_dict = construct_feed_dict(self.model, data, state, att_states, att_counts)
                results = session.run(evals, feed_dict=feed_dict)
                prediction, state, att_states, att_counts, att_vec, lambda_vec = \
                    extract_results(results, evals, 1, self.model)
                predicted = np.argmax(prediction)
                current_token = self.inverse_map[token]
                predicted_token = self.inverse_map[predicted]

                if len(accumulated_tokens) > self.max_attention:
                    accumulated_tokens.pop(0)

                m = att_vec[0].shape[1]
                take = min(m, len(accumulated_tokens))
                alphas = att_vec[0][0, m-take:]
                labels = np.array([clean_token(t) for t in accumulated_tokens])

                '''if take > self.max_display:
                    ind = np.argpartition(alphas, -self.max_display)[-self.max_display:]
                    alphas = alphas[ind]
                    labels = labels[ind]'''

                y_labels.append(current_token.replace("\n", "<newline>"))
                predicted_labels.append(predicted_token.replace("\n", "<newline>"))

                print("%s ; %s" % (current_token.replace("\n", "<newline>"), predicted_token.replace("\n", "<newline>")))

                begin = max(self.max_display-take, 0)
                plot_data[i, begin:] = alphas
                annotations[i, begin:] = labels
                if begin != 0:
                    annotations[i, 0:begin] = ""

                accumulated_tokens.append(current_token)

            for i in range(1, len(y_labels)):
                if y_labels[i] == predicted_labels[i-1]:
                    y_labels[i] = "** " + y_labels[i]

            x_labels = [""] * self.max_display

            sns.set(font_scale=1.2)
            sns.set_style({"savefig.dpi": 100})
            ax = sns.heatmap(plot_data, cmap=plt.cm.Blues, linewidths=.1, annot=annotations, fmt="", vmin=0, vmax=1,
                             cbar=False, xticklabels=x_labels, yticklabels=y_labels, annot_kws={"size": 10})
            plt.yticks(rotation=0)

            fig = ax.get_figure()
            # specify dimensions and save
            fig.set_size_inches(int(self.max_display)*1.3, int(len(plot_data)/3))

            fig.savefig('./out/lagged_attention.png')
            print("Generated file lagged_attention.png")


def bin(data, no_bin):
    return [int(round(d*no_bin)) for d in data]

'''
def bin(data, no_bins):
    bins = np.linspace(0, 1, no_bins)
    digitized = np.digitize(data, bins)
    return digitized
'''

all_test_cases = [
    #["import", "numpy", "as", "np", "\n", "(*) data", "=", "np", ".", "array", "(", "[", "§NUM§", ",", "§NUM§", ",", "§NUM§", ",", "§NUM§", ",", "]", ")", "\n", "np", "."],
    #["import", "numpy", "as", "np", "\n", "(*) size", "=", "[", "§NUM§", ",", "§NUM§", "]", "\n", "(*) var324", "=", "np", "."],
    #["def", "(*) function234", "(", "(*) arg289", ")", ":", "\n", "§<indent>§", "with", "open", "(", "§OOV§", ",", "'r'", ")", "as", "(*) var76", ":", "\n", "§<indent>§", "(*) var91", "=", "var76", "."],
    #["def", "(*) function234", "(", "(*) arg289", ")", ":", "\n", "§<indent>§", "with", "open", "(", "§OOV§", ",", "'w'", ")", "as", "(*) var76", ":", "\n", "§<indent>§", "var76", "."],
    #["class", "(*) Class234", ":", "\n", "§<indent>§", "def", "(*) function123", "(", "self", ",", "(*) arg645", ",", "(*) arg631", ")", ":", "\n", "§<indent>§", "(*) var209", "=", "arg645", "+", "arg631", "\n", "return"],
    #["class", "(*) Class234", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg123", ")", ":", "\n", "§<indent>§", "self", ".", "(*) attribute353", "=", "arg123", "\n", "§<dedent>§", "def", "(*) function123", "(", "self", ",", "(*) arg645", ")", ":", "\n", "§<indent>§", "return", "self", "."],
    #["def", "(*) function943", "(", "(*) arg153", ")", ":", "\n", "§<indent>§", "return", "§OOV§", "\n"],
    #["class", "(*) Class133", ":", "\n", "§<indent>§", "def"],
    #["class", "(*) Class133", ":", "\n", "§<indent>§", "def", "__init__"],
    #["class", "(*) Class133", ":", "\n", "§<indent>§", "def", "__init__", "("],
    #["class", "(*) Class133", ":", "\n", "§<indent>§", "def", "(*) function239", "("],
    #["def", "(*) function129", "("],
    #["class", "(*) Class23", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg932", ")", ":", "\n", "§<indent>§"],
    #["class", "(*) Class23", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg932", ")", ":", "\n", "§<indent>§", "self", "."],
    #["class", "(*) Class23", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg932", ")", ":", "\n", "§<indent>§", "self", ".", "(*) attribute453", "="],
    #["for"],
    #["for", "(*) var983", "in"],
    #["if"],
    #["from", "datetime", "import", "datetime", "\n", "(*) now", "=", "datetime", "."],
    #["for", "(*) var948", "in", "range", "(", "§NUM§", ")", ":", "\n", "§<indent>§", "if", "var948", "%", "§NUM§", "==", "§NUM§", ":", "\n", "§<indent>§"],
    #["import", "shutil", "\n", "def", "(*) function220", "(", "(*) arg287", ")", ":", "\n", "§<indent>§", "try", ":", "\n", "§<indent>§", "shutil", ".", "rmtree", "(", "arg287", ")", "\n", "§<dedent>§"],
    #["import", "shutil", "\n", "def", "(*) function220", "(", "(*) arg287", ")", ":", "\n", "§<indent>§", "try", ":", "\n", "§<indent>§", "shutil", ".", "rmtree", "("],
    #["def"],
    #["def", "function20", "(", "s|arg123", ")", ":", "\n", "§<indent>§", "print", "("],
    #["class", "MyClass|Class291", ":", "\n", "§<indent>§", "pass", "\n", "\n", "§<dedent>§", "var821", "="],
    #["def", "function20", "(", "arg123", ")", ":", "\n", "§<indent>§", "pass", "\n", "\n", "var821", "="],
    #["import", "os", "\n", "\n", "def", "function23", "(", "arg123", ")", ":", "\n", "§<indent>§", "os", ".", "path", "."],
    #["import", "os", "\n", "\n", "def", "function23", "(", "arg123", ")", ":", "\n", "§<indent>§", "os", ".", "path", ".", "join", "("],
    #["var464", "=", "[", "§NUM§", "§NUM§", "§NUM§", "]", "\n", "var921", "=", "sorted", "("],
    #["class", "MyClass|Class291", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "arg123", ")", ":", "\n", "§<indent>§", "self", ".", "attribute|attribute632", "=", "arg123", "\n", "\n", "§<dedent>§",
    # "def", "function234", "(", "self", ",", "filename|arg432", ")", ":", "\n", "§<indent>§", "with", "open", "(", "filename|arg432", ",", "'r'", ")", "as", "f|var76", ":", "\n", "§<indent>§", "lines|var91", "=", "f|var76", ".",
    # "readlines", "(", ")", "\n", "§<dedent>§", "return", "len", "(", "lines|var91", ")", "\n", "\n", "§<dedent>§", "def", "func|function921", "(", "self", ",", "arg1", ")", ":", "\n", "§<indent>§", "if", "self", "."]
    #["class", "Class210", ":", "\n", "\n", "§<indent>§", "def", "__init__", "(", "self", ",",
    # "arg233", ")", ":", "\n",
    # "§<indent>§",
    # "self", ".", "attribute|attribute172", "=", "arg233", "\n",
    # "\n", "§<dedent>§", "def", "function1234", "(", "self", ",", "arg635", ")", ":", "\n", "§<indent>§",
    # "return", "§OOV§", "if", "arg635", "else", "§OOV§", "\n",
    # "\n", "§<dedent>§", "def", "function651", "(", "self", ",", "arg536", ")", ":", "\n", "§<indent>§",
    # "return", "§OOV§", "if", "arg536", "else", "§OOV§", "\n",
    # "\n", "§<dedent>§", "def", "func|function2766", "(", "self", ",", "arg1|arg1556", ")", ":", "\n",
    # "§<indent>§", "var155", "=", "os", ".", "path", ".", "join", "(", "arg1|arg1556", ",", "self", "."]
    ["import", "os", "\n", "\n", "class", "Class253", ":", "\n", "\n", "§<indent>§",
     "def", "__init__", "(", "self", ",", "arg651", ")", ":", "\n", "§<indent>§",
     "self", ".", "attribute943", "=", "arg651", "\n", "\n", "§<dedent>§",
     "def", "function1690", "(", "self", ",", "arg2004", ")", ":", "\n", "§<indent>§",
     "var4040", "=", "os", ".", "path", ".", "join", "(", "self", ".", "attribute943", ",", "arg2004", ")", "\n",
     "print", "(", "§OOV§", "%", "(", "str", "("
    ],
    #["import", "os", "\n", "class", "(*) MyClass|Class291", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg123", ")", ":", "\n", "§<indent>§", "self", ".", "(*) attribute|attribute172", "=", "arg123", "\n", "\n", "§<dedent>§",
    # "def", "(*) function234", "(", "self", ",", "(*) filename|arg432", ")", ":", "\n", "§<indent>§", "with", "open", "(", "filename|arg432", ",", "'r'", ")", "as", "(*) f|var76", ":", "\n", "§<indent>§", "(*) lines|var91", "=", "f|var76", ".",
    # "readlines", "(", ")", "\n", "§<dedent>§", "return", "len", "(", "lines|var91", ")", "\n", "\n", "§<dedent>§", "def", "(*) func|function921", "(", "self", ",", "(*) arg191", ")", ":", "\n", "§<indent>§", "(*) var543", "=", "os", ".", "path", ".", "join", "(", "self", "."]
    ["class", "(*) Class210", ":", "\n", "\n", "§<indent>§", "def", "__init__", "(", "self", ",",
     "(*) arg233", ")", ":", "\n",
     "§<indent>§",
     "self", ".", "(*) attribute172", "=", "arg233", "\n",
     "\n", "§<dedent>§", "def", "(*) function1234", "(", "self", ",", "(*) arg635", ")", ":", "\n", "§<indent>§",
     "return", "§OOV§", "if", "arg635", "else", "§OOV§", "\n",
     "\n", "§<dedent>§", "def", "(*) function651", "(", "self", ",", "(*) arg536", ")", ":", "\n", "§<indent>§",
     "return", "§OOV§", "if", "arg536", "else", "§OOV§", "\n",
     "\n", "§<dedent>§", "def", "(*) function2766", "(", "self", ",", "(*) arg1556", ")", ":", "\n",
     "§<indent>§", "(*) var155", "=", "os", ".", "path", ".", "join", "(", "arg1556", ",", "self", "."],
    #["def", "(*) function20", "(", "(*) arg123", ")", ":", "\n", "§<indent>§", "print", "("],
    ["class", "(*) Class12", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ")", ":", "\n", "§<indent>§", "self", ".", "(*) attribute462", "=", "§OOV§", "\n", "\n", "§<dedent>§", "§<dedent>§", "if", "__name__", "==", "'__main__'", ":", "\n", "§<indent>§", "(*) var821", "="],
    ["def", "(*) function20", "(", "(*) arg123", ")", ":", "\n", "§<indent>§", "pass", "\n", "\n", "(*) var821", "="],
    ["import", "os", "\n", "\n", "def", "(*) function23", "(", "(*) arg123", ")", ":", "\n", "§<indent>§", "os", ".", "path", "."],
    ["import", "os", "\n", "\n", "def", "(*) function23", "(", "(*) arg123", ")", ":", "\n", "§<indent>§", "os", ".", "path", ".", "join", "("],
    ["(*) var464", "=", "[", "§NUM§", "§NUM§", "§NUM§", "]", "\n", "(*) var921", "=", "sorted", "("],
    ["import", "os", "\n", "class", "(*) Class291", ":", "\n", "§<indent>§", "def", "__init__", "(", "self", ",", "(*) arg123", ")", ":", "\n", "§<indent>§", "self", ".", "(*) attribute|attribute172", "=", "arg123", "\n", "\n", "§<dedent>§",
     "def", "(*) function234", "(", "self", ",", "(*) filename|arg432", ")", ":", "\n", "§<indent>§", "with", "open", "(", "filename|arg432", ",", "'r'", ")", "as", "(*) f|var76", ":", "\n", "§<indent>§", "(*) lines|var91", "=", "f|var76", ".",
     "readlines", "(", ")", "\n", "§<dedent>§", "return", "len", "(", "lines|var91", ")", "\n", "\n", "§<dedent>§", "def", "(*) func|function921", "(", "self", ",", "(*) arg191", ")", ":", "\n", "§<indent>§", "(*) var543", "=", "os", ".", "path", ".", "join", "(", "self", "."],
    #["class", "(*) Class210", ":", "\n", "\n", "§<indent>§", "def", "__init__", "(", "self", ",",
    # "(*) arg233", ")", ":", "\n",
    # "§<indent>§",
    # "self", ".", "(*) attribute172", "=", "arg233", "\n",
    # "\n", "§<dedent>§", "def", "(*) function1234", "(", "self", ",", "(*) arg635", ")", ":", "\n", "§<indent>§",
    # "return", "§OOV§", "if", "arg635", "else", "§OOV§", "\n",
    # "\n", "§<dedent>§", "def", "(*) function651", "(", "self", ",", "(*) arg536", ")", ":", "\n", "§<indent>§",
    # "return", "§OOV§", "if", "arg536", "else", "§OOV§", "\n",
    # "\n", "§<dedent>§", "def", "(*) function2766", "(", "self", ",", "(*) arg1556", ")", ":", "\n",
    # "§<indent>§", "(*) var155", "=", "os", ".", "path", ".", "join", "(", "arg1556", ",", "self", "."]

]


# TODO: FLAG IDENTIFIERS FOR THE ATTENTION MODEL

def map_token(map, token):
    mask = 0
    if token.startswith("(*) "):
        mask = 1
        token = token.replace("(*) ", "")

    if token in map:
        return map[token], mask

    # Not in map, is it an identifier?
    if "|" in token:
        spl = token.split("|")
        if spl[1] in map:
            return map[spl[1]]
        elif spl[0] in map:
            return map[spl[0]]

    elif any([token.startswith(prefix) for prefix in astwalker.possible_types()]):
        return pyreader.oov_id

    raise KeyError(token)


class TopPredictions:
    def __init__(self, model, map, attns):
        self.model = model
        self.map = map
        self.inv_map = {v: k for k, v in map.items()}
        self.attns = attns
        self.prediction_op = tf.nn.top_k(model.predict, 5)

    def __call__(self, session):
        to_eval = [self.prediction_op[0], self.prediction_op[1]]
        evals = get_evals(to_eval, self.model)
        for testcase in all_test_cases:
            state, att_states, att_counts = get_initial_state(self.model)
            for i, token in enumerate(testcase):
                att_mask = attention_masks(self.attns, [0], 1)
                data = (np.array([[ map_token(self.map, testcase[i]) ]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
                token_id, mask = map_token(self.map, testcase[i])
                att_mask = attention_masks(self.attns, np.array([mask]), 1)
                data = (np.array([[ token_id ]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
                feed_dict = construct_feed_dict(self.model, data, state, att_states, att_counts)

                results = session.run(evals, feed_dict)
                results, state, att_states, att_counts, _, _ = extract_results(results, evals, 2, self.model)

            probs = results[0]
            predict_ids = results[1]

            for i in range(5):
                print("%s ; %f" % (self.inv_map[predict_ids[0, i]].replace("\n", "<newline>"), probs[0, i]))

            print("\n\n§§§§§§§§§§§§§\n\n")


class BeamSearch:
    def __init__(self, model, map, attns, beam_width):
        self.model = model
        self.map = map
        self.inv_map = {v: k for k, v in map.items()}
        self.prediction_op = tf.nn.top_k(model.predict, beam_width)
        self.attns = attns
        self.beam_width = beam_width

    def __call__(self, session, depth):

        def is_identifier(token_id):
            token = self.inv_map[token_id]
            if any(token.startswith(p) for p in astwalker.possible_types()):
                return 1
            return 0

        def run_network(token_id, mask, state, att_states, att_ids, att_counts):
            att_mask = attention_masks(self.attns, np.array([mask]), 1)
            data = (np.array([[token_id]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
            feed_dict = construct_feed_dict(self.model, data, state, att_states, att_ids, att_counts)

            results = session.run(evals, feed_dict)
            results, state, att_states, att_ids, alpha_states, att_counts, lambda_state = extract_results(results, evals, 2, self.model)
            return results, state, att_states, att_ids, att_counts

        def beam(tree_node):
            # Populate the children of tree_node
            init_state, init_att_states, init_att_ids, init_att_counts = tree_node.state
            results, state, att_states, att_ids, att_counts = run_network(tree_node.token_id, tree_node.mask, init_state, init_att_states, init_att_ids, init_att_counts)
            probs = results[0]
            predict_ids = results[1]
            for i in range(self.beam_width):
                tree_node.add_child(BeamSearchTreeNode(predict_ids[0, i], is_identifier(predict_ids[0, i]), (state, att_states, att_ids, att_counts), probs[0, i]))

        def beam_search_recursive(tree, current_depth):
            if current_depth < depth:
                for child in tree.children:
                    beam(child)
                    beam_search_recursive(child, current_depth+1)

        to_eval = [self.prediction_op[0], self.prediction_op[1]]
        evals = get_evals(to_eval, self.model)
        for testcase in all_test_cases:
            # Pass the context through the network

            state, att_states, att_ids, att_counts = get_initial_state(self.model)
            for i, token in enumerate(testcase[:-1]):
                token_id, mask = map_token(self.map, token)
                results, state, att_states, att_ids, att_counts = run_network(token_id, mask, state, att_states, att_ids, att_counts)
                # print([self.inv_map[id] for id in att_ids[0].tolist()[0]])

            token_id, mask = map_token(self.map, testcase[-1])
            root = BeamSearchTreeNode(token_id, mask, (state, att_states, att_ids, att_counts), 1)

            beam(root)
            beam_search_recursive(root, 1)
            path = find_path(root)[0]


            print(" ".join([self.inv_map[map_token(self.map, t)[0]] for t in testcase]))

            for child in sorted(root.children, key=lambda c: c.probability, reverse=True):
                print("%s ; %f" % (self.inv_map[child.token_id].replace("\n", "<newline>"), child.probability))
            print()
            print(" ".join([self.inv_map[t].replace("\n", "<newline>") for t in path]))
            print("\n\n§§§§§§§§§§§§§§\n\n")



class BeamSearch2:
    def __init__(self, model, map, attns, beam_width):
        self.model = model
        self.map = map
        self.inv_map = {v: k for k, v in map.items()}
        self.prediction_op = tf.nn.top_k(model.predict, beam_width)
        self.attns = attns
        self.beam_width = beam_width

    def __call__(self, session, depth):
        def run_network(token_id, state, att_states, att_counts):
            att_mask = attention_masks(self.attns, [0], 1)
            data = (np.array([[token_id]]), np.array([[1]]), np.array([att_mask]), np.array([1]))
            feed_dict = construct_feed_dict(self.model, data, state, att_states, att_counts)

            results = session.run(evals, feed_dict)
            results, state, att_states, att_counts, _, _ = extract_results(results, evals, 2, self.model)
            return results, state, att_states, att_counts

        def beam(tree_node):
            # Populate the children of tree_node
            init_state, init_att_states, init_att_counts = tree_node.state
            results, state, att_states, att_counts = run_network(tree_node.token_id, init_state, init_att_states, init_att_counts)
            probs = results[0]
            predict_ids = results[1]
            for i in range(self.beam_width):
                tree_node.add_child(BeamSearchTreeNode(predict_ids[0, i], (state, att_states, att_counts), probs[0, i]))

        def beam_search_recursive(tree, current_depth):
            if current_depth < depth:
                for child in tree.children:
                    beam(child)
                    beam_search_recursive(child, current_depth+1)

        to_eval = [self.prediction_op[0], self.prediction_op[1]]
        evals = get_evals(to_eval, self.model)
        count = 0
        accurate = 0

        for testcase in all_test_cases:
            state, att_states, att_counts = get_initial_state(self.model)

            for i, token in enumerate(testcase[:-depth]):
                results, state, att_states, att_counts = run_network(map_token(self.map, token), state, att_states, att_counts)

                root = BeamSearchTreeNode(map_token(self.map, testcase[-1]), (state, att_states, att_counts), 1)
                beam(root)
                beam_search_recursive(root, 1)
                path = find_path(root)[0]  # The most likely path
                actual = [map_token(self.map, t) for t in testcase[i+1:i+depth+1]]

                # print("Token: %s" % token)
                # print("Predicted:")
                # print(" ".join([self.inv_map[t].replace("\n", "<newline>") for t in path]))
                # print("Actual:")
                # print(" ".join([self.inv_map[t].replace("\n", "<newline>") for t in actual]))
                # print("\n")

                count += 1
                if path == actual:
                    accurate += 1

        print("Accuracy: %f" % (accurate/count))

