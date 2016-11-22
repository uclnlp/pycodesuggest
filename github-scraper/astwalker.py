import itertools
import astor
import ast
import random


def get_position(node, offset=0):
    return node.lineno, node.col_offset + offset


type_max_rands = {
    "var": 4750, "function": 2900, "Class": 440, "attribute": 2400, "arg": 2400
}


def possible_identifiers():
    return [key + str(i) for key in type_max_rands for i in range(type_max_rands[key])]


def possible_types():
    return [key for key in type_max_rands]


class ASTWalker(astor.TreeWalk):
    def init_lists(self):
        self.ignore_names = ["self"]
        self._names = {}
        self._counters = {}
        self.scope_variables = ["function", "arg", "attribute", "var", "Class"]
        self.class_variables = ["function", "attribute"]
        self.class_type = "Class"
        self.scope = ()
        self.class_scope = ()
        self.class_map = {}  # Maps a class to its scope
        self.linked_class_scopes = {}
        self.variable_class_map = {}
        self.queue = []
        self.assign = False
        self.randomise = True
        self.allocated_random = {}
        self.definition_positions = []
        self.name_mapping = {}

    def begin_scope(self, scope_name, class_scope=False):
        self.scope = self.scope + (scope_name,)
        if class_scope:
            self.class_scope = self.class_scope + ("Class" + scope_name,)

    def end_scope(self, class_scope=False):
        self.scope = self.scope[0:-1]
        if class_scope:
            self.class_scope = self.class_scope[0:-1]

    def define_name(self, typename, name, position, scope=None, typenames=None, add_to_scopes=[]):
        if scope is None:
            scope = self.scope

        if typenames is None:
            typenames = self.scope_variables

        new_name = self.lookup_name(scope, typenames, name)
        if new_name is not None:
            return new_name

        identifier = self.get_random(typename) if self.randomise else self.get_counter(typename, scope)
        new_name = typename + str(identifier)
        self.add_name_mapping(scope, typename, name, new_name, position)
        self.definition_positions.append((typename, position))

        for scope in add_to_scopes:
            self.add_name_mapping(scope, typename, name, new_name, position)

        return new_name

    def add_name_mapping(self, scope, typename, name, new_name, position):
        if scope not in self._names:
            self._names[scope] = {}
        if typename not in self._names[scope]:
            self._names[scope][typename] = {}
        self._names[scope][typename][name] = new_name
        self.name_mapping[new_name] = position

    def get_random(self, typename):
        if typename not in self.allocated_random:
            self.allocated_random[typename] = []

        max_rand = type_max_rands[typename]
        # Running out of random numbers to allocate, so instead do a linear search of unallocated numbers
        if len(self.allocated_random[typename]) >= (max_rand // 4) * 3:
            value = 0
            while value not in self.allocated_random[typename]:
                value += 1
        else:
            value = random.randrange(max_rand)
            while value in self.allocated_random[typename]:
                value = random.randrange(max_rand)

        self.allocated_random[typename].append(value)
        return value

    def get_counter(self, typename, scope=None):
        if scope is None:
            scope = self.scope_variables

        counter = 0
        for s in [scope[:s] for s in reversed(range(len(scope) + 1))]:
            if s in self._counters and typename in self._counters[s]:
                counter = self._counters[s][typename]
                break

        counter += 1
        if scope not in self._counters:
            self._counters[scope] = {}

        self._counters[scope][typename] = counter
        return counter

    def lookup_name(self, lookup_scope, typenames, name):
        for scope in reversed([lookup_scope[:s] for s in range(len(lookup_scope) + 1)]):
            if scope in self._names:
                for typename in typenames:
                    if typename in self._names[scope] and name in self._names[scope][typename]:
                        return self._names[scope][typename][name]

        return None

    def lookup_class_scope(self, scope, name):
        for s in [scope[:s] for s in range(len(scope) + 1)]:
            if s in self.variable_class_map:
                if name in self.variable_class_map[s]:
                    return self.variable_class_map[s][name]

        return None

    def replace_name(self, name, target_name):
        if target_name is not None:
            if isinstance(name, ast.Name):
                name.id = target_name
            elif isinstance(name, ast.Attribute):
                name.attr = target_name

    def pre_ClassDef(self):
        def add_linked_class_scope(class_name, linked):
            if class_name not in self.linked_class_scopes:
                self.linked_class_scopes[class_name] = []

            self.linked_class_scopes[class_name].append(linked)

        new_name = self.define_name("Class", self.cur_node.name, get_position(self.cur_node, 6))
        self.cur_node.name = new_name
        self.begin_scope(self.cur_node.name, class_scope=True)
        self.class_map[new_name] = self.class_scope
        if hasattr(self.cur_node, "bases"):  # Inheritance
            for base in [b for b in self.cur_node.bases if isinstance(b, ast.Name)]:
                self.check_and_replace(base, [self.class_type])
                parent_class = self.lookup_name(self.scope, [self.class_type], base.id)
                if parent_class is not None:
                    add_linked_class_scope(self.class_map[parent_class], self.class_map[new_name])
                    add_linked_class_scope(self.class_map[new_name], self.class_map[parent_class])

    def post_ClassDef(self):
        self.end_scope(class_scope=True)

    def pre_ExceptHandler(self):
        self.begin_scope("Exception")

        def get_len(node):
            if isinstance(node, ast.Name):
                return len(node.id)
            elif isinstance(node, ast.Attribute):
                return len(node.value.id) + len(node.attr) + 1
            else:
                return 0

        if hasattr(self.cur_node, "name") and self.cur_node.name:
            offset = 11
            if isinstance(self.cur_node.type, ast.Tuple):
                offset += sum([get_len(n) + 2 for n in self.cur_node.type.elts])
            else:
                offset += get_len(self.cur_node)

            new_name = self.define_name("var", self.cur_node.name, get_position(self.cur_node, offset))
            self.cur_node.name = new_name

    def post_ExceptionHandler(self):
        self.end_scope()

    def pre_FunctionDef(self):
        name = self.cur_node.name
        if not name.startswith("__"):
            new_name = self.define_name("function", name, get_position(self.cur_node, 4),
                                        add_to_scopes=[self.class_scope])
            self.cur_node.name = new_name

        self.begin_scope(name)

    def post_FunctionDef(self):
        self.end_scope()

    def pre_ListComp(self):
        self.begin_scope("listcomp")

    def post_ListComp(self):
        self.check_and_replace(self.cur_node.elt)
        self.end_scope()

    def process_assign_target(self, target):
        all_targets = []
        if isinstance(target, ast.Name):
            all_targets.append(target)

        if isinstance(target, ast.Tuple):
            all_targets.extend(x for x in target.elts)

        for target in all_targets:
            if isinstance(target, ast.Name):
                new_name = self.define_name("var", target.id, get_position(target))
                target.id = new_name
            elif isinstance(target, ast.List):
                for name in target.elts:
                    new_name = self.define_name("var", name.id, get_position(name))
                    name.id = new_name

    def pre_comprehension(self):
        self.process_assign_target(self.cur_node.target)
        self.check_and_replace(self.cur_node.iter)

    def pre_arguments(self):
        for arg in self.cur_node.args:
            if arg.arg not in self.ignore_names:
                new_name = self.define_name("arg", arg.arg, get_position(arg))
                arg.arg = new_name

    def pre_Assign(self):
        all_targets = list(itertools.chain(
            [x for x in self.cur_node.targets if isinstance(x, ast.Name)],
            [y for x in self.cur_node.targets if isinstance(x, ast.Tuple) for y in x.elts if isinstance(y, ast.Name)]
        ))

        for target in all_targets:
            new_name = self.define_name("var", target.id, get_position(target))
            target.id = new_name

        self.check_and_replace(self.cur_node.value)

        # If a variable is being assigned to a class, we need to keep track of it
        # so that attributes can later refer to the relevant scope for that class
        if isinstance(self.cur_node.value, ast.Call) and isinstance(self.cur_node.value.func, ast.Name):
            class_name = self.lookup_name(self.scope, [self.class_type], self.cur_node.value.func.id)
            if class_name is not None:
                if self.scope not in self.variable_class_map:
                    self.variable_class_map[self.scope] = {}

                for target in all_targets:
                    self.variable_class_map[self.scope][target.id] = self.class_map[class_name]

    def pre_Tuple(self):
        for elt in self.cur_node.elts:
            self.check_and_replace(elt)

    def check_and_replace(self, candidate, typenames=None, scope=None, class_scope=False):
        if typenames is None:
            typenames = self.scope_variables
        if scope is None:
            scope = self.scope

        if isinstance(candidate, ast.Name) or isinstance(candidate, ast.Attribute):
            self.queue.append((scope, typenames, candidate, class_scope))
        elif hasattr(candidate, "elts"):
            for elt in candidate.elts:
                self.check_and_replace(elt, typenames=typenames, scope=scope)
        elif isinstance(candidate, ast.Starred):
            self.check_and_replace(candidate.value, typenames=typenames, scope=scope)

    def process_replace_queue(self):
        for scope, typenames, candidate, class_scope in self.queue:
            name = candidate.id if isinstance(candidate, ast.Name) else candidate.attr
            target_name = self.lookup_name(scope, typenames, name)
            if target_name is None and class_scope:
                for s in self.linked_class_scopes.get(scope, []):
                    target_name = self.lookup_name(s, typenames, name)

            self.replace_name(candidate, target_name)

    def pre_BinOp(self):
        self.check_and_replace(self.cur_node.left)
        self.check_and_replace(self.cur_node.right)

    def pre_UnaryOp(self):
        self.check_and_replace(self.cur_node.operand)

    def pre_Subscript(self):
        self.check_and_replace(self.cur_node.value)

    def pre_Index(self):
        self.check_and_replace(self.cur_node.value)

    def pre_AugAssign(self):
        self.check_and_replace(self.cur_node.target)
        self.check_and_replace(self.cur_node.value)

    def pre_Attribute(self):
        is_assigning = "targets" in self.parent_name
        is_self = isinstance(self.cur_node.value, ast.Name) and self.cur_node.value.id == "self"
        if is_self:
            if is_assigning:
                new_name = self.define_name("attribute", self.cur_node.attr, get_position(self.cur_node, 5),
                                            scope=self.class_scope, typenames=self.class_variables)
                self.cur_node.attr = new_name
            else:
                self.check_and_replace(self.cur_node, typenames=self.class_variables,
                                       scope=self.class_scope, class_scope=True)

        else:
            self.check_and_replace(self.cur_node.value)

        # Now handle attributes on other classes
        if isinstance(self.cur_node.value, ast.Name):
            target_name = self.lookup_name(self.scope, self.scope_variables, self.cur_node.value.id)
            if target_name is not None:
                class_scope = self.lookup_class_scope(self.scope, target_name)
                if class_scope is not None:
                    self.check_and_replace(self.cur_node, typenames=self.class_variables, scope=class_scope)

    def pre_Slice(self):
        self.check_and_replace(self.cur_node.lower)
        self.check_and_replace(self.cur_node.upper)

    def pre_Compare(self):
        self.check_and_replace(self.cur_node.left)
        for comparator in self.cur_node.comparators:
            self.check_and_replace(comparator)

    def pre_Return(self):
        self.check_and_replace(self.cur_node.value)

    def pre_withitem(self):
        if hasattr(self.cur_node, "optional_vars") and isinstance(self.cur_node.optional_vars, ast.Name):
            new_name = self.define_name("var", self.cur_node.optional_vars.id,
                                        get_position(self.cur_node.optional_vars))
            self.cur_node.optional_vars.id = new_name

    def pre_For(self):
        self.begin_scope("For")
        self.process_assign_target(self.cur_node.target)
        self.check_and_replace(self.cur_node.iter)

    def post_For(self):
        self.end_scope()

    def pre_If(self):
        self.check_and_replace(self.cur_node.test)
        self.check_and_replace(self.cur_node.body)
        self.check_and_replace(self.cur_node.orelse)

    def pre_IfExp(self):
        self.pre_If()

    def pre_Call(self):
        if isinstance(self.cur_node.func, ast.Attribute):
            self.check_and_replace(self.cur_node.func.value)

        self.check_and_replace(self.cur_node.func)

        for arg in self.cur_node.args:
            self.check_and_replace(arg)

    def pre_Name(self):
        pass

    def pre_keyword(self):
        self.check_and_replace(self.cur_node.value)

    def print(self):
        print("Identifiers:\n%s" % "\n".join(
            [x + "\t" + y for z in self.names for y in self.names[z] for x in self.names[z][y]]))

    @property
    def names(self):
        return self._names

    @property
    def name_usage(self):
        def position(node):
            def offset(attr):
                if isinstance(attr.value, ast.Attribute):
                    return len(attr.attr) + offset(attr.value) + 1
                elif isinstance(attr.value, ast.Name):
                    return len(attr.value.id) + 1
                else:
                    return 0

            return get_position(node) if isinstance(node, ast.Name) \
                else get_position(node, offset(node))

        def name(node):
            return node.id if isinstance(node, ast.Name) else node.attr

        def_positions = [p[1] for p in self.definition_positions]
        identifier_names = [k3 for k in self.names for k2 in self.names[k] for k3 in self.names[k][k2]]
        return [position(candidate) for _, _, candidate, _ in self.queue
                if position(candidate) not in def_positions and
                name(candidate) in identifier_names]


if __name__ == "__main__":

    def get_source_tree(filename):
        with open(filename, 'r') as f:
            fstr = f.read()
        fstr = fstr.replace('\r\n', '\n').replace('\r', '\n')
        if not fstr.endswith('\n'):
            fstr += '\n'
        return fstr, ast.parse(fstr, filename=filename)

    file = "/Users/avishkar/pythonRepos_vars_format/mher/flower/pavement.py"
    # file = "/Users/avishkar/pyRepos/debug/PyRepo.py"
    source, tree = get_source_tree(file)
    walker = ASTWalker(tree)
    print(walker.names)
    print(walker.name_usage)

    # print(astor.dump_tree(tree))
