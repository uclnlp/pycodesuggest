class BeamSearchTreeNode(object):

    def __init__(self, token_id, mask, state, probability):
        self._token_id = token_id
        self._state = state
        self._probability = probability
        self._children = []
        self._mask = mask

    @property
    def mask(self):
        return self._mask

    @property
    def token_id(self):
        return self._token_id

    @property
    def state(self):
        return self._state

    @property
    def probability(self):
        return self._probability

    @property
    def children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)

