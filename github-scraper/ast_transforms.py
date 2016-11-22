import random


class ASTTransformer:
    def __init__(self, randomise):
        self.mappings = {}
        self.inverse_mappings = {}
        self.randomise = randomise

    def define_name(self, node, typename):
        scope = node.parent.scope()
        if scope not in self.mappings:
            self.mappings[scope] = {}
        if scope not in self.inverse_mappings:
            self.inverse_mappings[scope] = {}

        self.mappings[scope][node.name] = typename

    def generate_name(self, node, scope, typename):
        def generate_random():
            name = typename + str(random.randrange(1000))
            while node.scope_lookup(node, name)[1]:
                name = typename + str(random.randrange(1000))
            return name

        def generate_sequential():
            counter = 1
            name = typename + str(counter)
            while self.lookup_mapping(scope, name):
                counter += 1
                name = typename + str(counter)
            return name

        return generate_random() if self.randomise else generate_sequential()

    def lookup_mapping(self, scope, name):
        while scope:
            if scope in self.inverse_mappings and name in self.inverse_mappings[scope]:
                return self.inverse_mappings[scope][name]
            scope = scope.parent

    def transform_module(self, m):
        pass

    def transform_classdef(self, c):
        self.define_name(c, "Class")
