from anytree import Node, RenderTree
import operator
import pandas as pd


class ParsedDecisionTreeClassifier:
    """
    Important:
        1. There will be a mismatch in the predictions sometimes.
        2. Make sure max_depth does not exceed when exporting the tree.
    """

    def __init__(self, tree_text, feature_names):
        self.tree_text = tree_text
        self.feature_names = feature_names
        self.root_node = self.build_tree()

    def build_tree(self):
        root_node = Node("root")
        stack = [root_node]
        lines = [ln for ln in self.tree_text.split("\n") if ln.strip() != ""]
        for line in lines:
            depth = 1
            while line.startswith("|   " * depth):
                depth += 1
            prefix = "|   " * (depth - 1) + "|--- "
            content = line[len(prefix):].strip()
            if content.startswith("class:"):
                class_label = content.split("class:")[1].strip()
                node = Node(content, depth=depth, class_label=class_label)
            else:
                op_str = "<=" if "<=" in content else ">"
                feature_name = content.split(op_str)[0].strip()
                threshold = float(content.split(op_str)[1].strip())
                op_func = operator.le if op_str == "<=" else operator.gt
                if feature_name not in self.feature_names:
                    raise ValueError(f"Feature '{feature_name}' not found in feature_names.")
                node = Node(content,
                            depth=depth,
                            feature_idx=self.feature_names.index(feature_name),
                            threshold=threshold,
                            op_func=op_func)
            while len(stack) > depth:
                stack.pop()
            if stack:
                node.parent = stack[-1]
            stack.append(node)
        return root_node

    def evaluate_tree(self, node, sample):
        if hasattr(node.children[0], "class_label"):
            return node.children[0].class_label
        feature_idx = node.children[0].feature_idx
        threshold = node.children[0].threshold
        op_func = node.children[0].op_func
        if op_func(sample[feature_idx], threshold):
            return self.evaluate_tree(node.children[0], sample)
        else:
            return self.evaluate_tree(node.children[1], sample)

    def validate_tree(self, node=None):
        if node is None:
            node = self.root_node
        if len(node.children) == 0:
            raise ValueError(f"Node '{node.name}' is a leaf node without a class_label.")
        elif len(node.children) == 1:
            if hasattr(node.children[0], "class_label"):
                return True
        if len(node.children) == 2:
            for child in node.children:
                if hasattr(child, "class_label"):
                    raise ValueError(f"Node '{node.name}' has a child with a class_label; but is not a leaf node.")
                if self.validate_tree(child):
                    continue
            return True
        raise ValueError(f"Node '{node.name}' does not have 2 children. "
                         f"The opposite of the child node '{node.children[0].name}' cannot be found.")

    def get_tree_text(self):
        tree_text = ""
        for pre, fill, node in RenderTree(self.root_node):
            tree_text += f"{pre}{node.name}\n"
        return tree_text

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.tolist()
        return [self.evaluate_tree(self.root_node, sample) for sample in X]
