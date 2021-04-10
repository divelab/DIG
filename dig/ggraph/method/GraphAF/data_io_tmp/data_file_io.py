import numpy as np
import rdkit.Chem as Chem
import csv
import pickle
import networkx as nx

def read_smile_file(path):
    fp = open(path, 'r')
    smile_list = []
    for smile in fp:
        smile = smile.strip()
        smile_list.append(smile)
    fp.close()
    return smile_list

def read_smile_prop_file(path):
    smile_list, prop_list = [], []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[0]]
            prop = row[columns[1]]
            smile_list.append(smile)
            prop_list.append(prop)
    return smile_list, prop_list

def read_com_file(path):
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    new_graphs = []
    for graph in graphs:
        self_loops = list(nx.selfloop_edges(graph))
        if len(self_loops) > 0:
            graph.remove_edges_from(self_loops)
        new_graph = nx.convert_node_labels_to_integers(graph)
        new_graphs.append(new_graph)
    return new_graphs