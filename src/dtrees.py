# author: Andre Hoffmann

import pandas as pd
import random
import scipy.stats
import math
import subprocess
import os


class Tree:
    def __init__(self, name, key, content, parent=None):
        self.name = name  # name of current attribute/value
        self.key = key
        self.content = content  # dataframe of dataset
        self.values = get_stats(content, key)  # counts of the key attribute values
        self.parent = parent
        self.children = []
        self.class_label = None
        self.color = 'black'

    def set_children(self, children):
        self.children = children

    def set_class_label(self, class_label):
        self.class_label = class_label
        if self.class_label == 'yes':
            self.color = 'green'
        elif self.class_label == 'no':
            self.color = 'red'

    def is_pure(self):
        return not bool(get_entropy(self.content[self.key]))

    # return class label for data point x as predicted by decision-tree dtree, where x is dataframe containing one row
    def classify(self, x):
        if self.is_pure():
            return self.class_label
        elif len(self.children) == 1:
            return self.children[0].classify(x)
        else:
            go_next = x[self.name].iloc[0]
            child_names = list(map(lambda n: n.name, self.children))
            next_node = self.children[child_names.index(go_next)]
            return next_node.classify(x)


def get_entropy(data):  # returns entropy for the given dataframe
    counts = data.value_counts()
    return scipy.stats.entropy(counts)


def get_infogain(root, attribute, key):
    subsets = []
    for val in root.content[attribute].unique():
        subsets.append(root.content[root.content[attribute] == val])
    summands = []
    for s in subsets:
        summands.append(float(len(s) / len(root.content)) * get_entropy(s[key]))
    infogain = get_entropy(root.content[key]) - sum(summands)
    return infogain


def get_iv(root, attribute):
    subsets = []
    for val in root.content[attribute].unique():
        subsets.append(root.content[root.content[attribute] == val])
    summands = []
    for s in subsets:
        summands.append(float(len(s) / len(root.content)) * math.log(float(len(s) / len(root.content)), 2))
    iv = - sum(summands)
    return iv


def get_gainratio(root, attribute, key):
    if get_iv(root, attribute) == 0:
        return 0
    gainratio = float(get_infogain(root, attribute, key) / (get_iv(root, attribute)))
    return gainratio


def get_split_attr(root, data, key, metric='gainratio'):
    attributes = data.columns.values.tolist()
    gains = []
    for a in attributes:
        if metric == 'infogain':
            gains.append(get_infogain(root, a, key))
        elif metric == 'gainratio':
            gains.append(get_gainratio(root, a, key))
        else:
            print("specify a valid metric!")
            return None
    return attributes[gains.index(max(gains))]  # return attribut maximizing information gain


# takes a data set and a key attribute, returns a dictionary of
# the counts for each attribute value of the attribute to classify on
def get_stats(data, key):
    counts = data[key].value_counts()
    stats = dict.fromkeys(list(data[key].unique()))
    for i in stats:
        stats[i] = counts[i]
    if 'yes' in stats and 'no' not in stats:
        stats['no'] = 0
    if 'no' in stats and 'yes' not in stats:
        stats['yes'] = 0
    if 'Yes' in stats and 'No' not in stats:
        stats['No'] = 0
    if 'No' in stats and 'Yes' not in stats:
        stats['Yes'] = 0
    return stats


def split(root, key):
    # determine split-attribute
    split_data = root.content.drop([key], axis=1)
    split_attr = get_split_attr(root, split_data, key)
    # create attribute-node (do i need this?)
    new_node = Tree(name=split_attr, key=key, content=root.content, parent=root)
    root.children.append(new_node)
    # create one child for each possible split-attribute value
    # where each data point gets assigned to the child named after the attr-value that matches the data point
    for val in new_node.content[split_attr].unique():
        new_node.children.append(Tree(name=val, key=key, content=new_node.content[new_node.content[split_attr] == val]))
    # if a node is not pure then recursively keep splitting
    # else node now represents the corresponding class
    for child in new_node.children:
        if child.is_pure():
            child.set_class_label(child.content[key].iloc[0])
        else:
            split(child, key)


def build_decision_tree(tdata, key_attr):
    root = Tree(name=None, key=key_attr, content=tdata)
    split(root, key_attr)
    return root


# create the code to visualize a decision tree in latex.
# nodes is a list of Tree objects
def make_body(nodes):
    if nodes is []:
        return ''
    elif len(nodes) == 1:
        if nodes[0].name is None:  # bogus root: skip
            return make_body(nodes[0].children)
        else:
            my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{tup1} }}{{\textcolor{{red}}{tup2} }} , ellipse,  draw {kids}]'.format(
                name=nodes[0].name, tup1=nodes[0].values['yes'], tup2=nodes[0].values['no'],
                kids=make_body(nodes[0].children))
            return my_string
    else:
        strings = []
        for node in nodes:
            if len(node.values) == 1:
                if 'yes' in node.values:
                    my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{tup1} }}{{\textcolor{{red}}{{0}} }} , rectangle,  draw={col} {kids}]'.format(
                        name=node.name, tup1=node.values[list(node.values.keys())[0]], kids=make_body(node.children), col=node.color)
                    strings.append(my_string)
                elif 'no' in node.values:
                    my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{{0}} }}{{\textcolor{{red}}{tup2} }} , rectangle,  draw={col} {kids}]'.format(
                        name=node.name, tup2=node.values[list(node.values.keys())[0]], kids=make_body(node.children), col=node.color)
                    strings.append(my_string)
                elif 'No' in node.values:
                    my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{{0}} }}{{\textcolor{{red}}{tup2} }} , rectangle,  draw={col} {kids}]'.format(
                        name=node.name, tup2=node.values[list(node.values.keys())[0]], kids=make_body(node.children), col=node.color)
                    strings.append(my_string)
                elif 'Yes' in node.values:
                    my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{tup1} }}{{\textcolor{{red}}{{0}} }} , rectangle,  draw={col} {kids}]'.format(
                        name=node.name, tup1=node.values[list(node.values.keys())[0]], kids=make_body(node.children), col=node.color)
                    strings.append(my_string)
            else:
                my_string = r'[{name}\\ \nicefrac{{\textcolor{{green}}{tup1} }}{{\textcolor{{red}}{tup2} }} , rectangle,  draw={col} {kids}]'.format(
                    name=node.name, tup1=node.values['yes'], tup2=node.values['no'], kids=make_body(node.children), col=node.color)
                strings.append(my_string)
        return ''.join(strings)


def visualize_tree(root):
    header = r'''
    \documentclass{article}
    \usepackage{nicefrac}
    \usepackage{tikz,forest}
    \usepackage{color} 
    \usetikzlibrary{arrows.meta}
    \forestset{qtree/.style={for tree={align=center, }}}
    \begin{document}
    \begin{center}
    \pgfkeys{/pgf/inner sep=0.6666em}
    \begin{forest}, baseline, qtree
    '''
    footer = '''
    \end{forest}
    \end{center}
    \end{document}
    '''
    body = make_body([root])
    content = header + body + footer
    with open('dtree.tex', 'w') as f:
        f.write(content)
    commandline = subprocess.Popen(['pdfLatex', 'dtree.tex'])
    commandline.communicate()
    os.unlink('dtree.tex')
    os.unlink('dtree.log')
    os.unlink('dtree.aux')


############
if __name__ == "__main__":
    df = pd.read_csv("weather2.csv")
    tree = build_decision_tree(df, 'play')
    visualize_tree(tree)