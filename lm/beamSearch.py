import numpy as np
from BeamSearchTree import BeamSearchTreeNode


def best_k(np_array, k):
    indices = np.argpartition(np_array, -k)[-k:]
    return indices[np.argsort(np_array[indices])]


def find_path(tree, k=1):

    paths = []

    def search_tree(path, node, path_prob):
        if not node.children:
            paths.append((path, path_prob))
        else:
            for child in node.children:
                search_tree(path + [child.token_id], child, path_prob + np.log(child.probability))

    search_tree([], tree, 0)

    sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True)
    return [path[0] for path in sorted_paths[:k]]


class BeamSearch:

    def __init__(self, model, beam_width, depth):
        self.model = model
        self.beam_width = beam_width
        self.depth = depth

    def beam_search(self, session, token_id, state):
        return self.beam_search_k(session, token_id, state, 1)[0]

    def beam_search_k(self, session, token_id, state, k):
        root = BeamSearchTreeNode(token_id, state, 1)
        tree = self.beam_search_tree(session, root)
        paths = find_path(tree, k)
        return paths

    def beam_search_tree(self, session, root):

        def beam(tree_node):
            feed_dict = {
                self.model.input_data: np.array([np.array([tree_node.token_id])]),
                self.model.initial_state: tree_node.state
            }

            probabilities, state = session.run([self.model.predict, self.model.final_state], feed_dict)
            best_k_indices = best_k(probabilities[0], self.beam_width)
            for token_idx in best_k_indices:
                probability = probabilities[0][token_idx]
                tree_node.add_child(BeamSearchTreeNode(token_idx, state, probability))

        def beam_search_recursive(tree, current_depth):
            if current_depth < self.depth:
                for child in tree.children:
                    beam(child)
                    beam_search_recursive(child, current_depth+1)

        beam(root)
        beam_search_recursive(root, 1)
        return root

