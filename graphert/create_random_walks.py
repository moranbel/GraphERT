import itertools
import pickle
from os.path import join

from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from graphert.processing_data import load_dataset


def create_random_walks(graphs: dict, ps: list, qs: list, walk_lengths: list, num_walks_list: list,
                        dataset_name: str, filter_cc_nodes:list=None):
    '''
    Create pd.Dataframe with random walks as sentences 'sent', and time step as 'label'.
    Creating the corpus per combination of walk_length (sentence length) and num walks starting from each one of the nodes.
    In same corpus we will have different p and q parameters in order to traverse the graph in multiple options.

    :param graphs: Graph per time step. dictionary. key- time step, value- nx.Graph
    :param ps: list of p (the return parameter) values.
    :param qs: list of q (the in-out parameter) values.
    :param walk_lengths: list of walk lengths.
    :param num_walks_list: list of the number of random walks starting from each one of the nodes.
    :param dataset_name: name of the dataset
    :param filter_cc_nodes: provide list of nodes to filter if we want to exclude nodes not in the biggest connected component.
    :return:
    '''
    for walk_len in walk_lengths:
        for num_walks in num_walks_list:
            print(f"walk_len={walk_len}, num_walks={num_walks}")
            file_path = f'datasets/{dataset_name}/paths_walk_len_{walk_len}_num_walks_{num_walks}.csv'
            data_df_list = []
            nodes = set()
            p_q_pairs = list(itertools.product(ps, qs))
            for i, (time, graph) in enumerate(list(graphs.items())):
                print(time)
                for (p, q) in tqdm(p_q_pairs, total=len(p_q_pairs)):
                    if filter_cc_nodes:
                        graph.remove_nodes_from(
                            [node for node in graph if node not in filter_cc_nodes])  # remove nodes not in the biggest cc
                    graph = graph.to_undirected()
                    nodes = nodes.union(graph.nodes())
                    n2v_model = Node2Vec(graph, num_walks=num_walks, walk_length=walk_len, p=p, q=q, workers=4,
                                         quiet=True)
                    sents = [" ".join(sent) for sent in n2v_model.walks]
                    data_df_list.append(
                        pd.DataFrame(np.array([sents, [time] * len(sents), [p] * len(sents), [q] * len(sents)]).T,
                                     columns=['sent', 'time', 'p', 'q']))

            data_df = pd.concat(data_df_list)
            data_df.to_csv(file_path)


if __name__ == '__main__':
    # create corpus
    # sent, label
    # load facebook
    dataset_name = 'facebook'
    graph_path = 'data/facebook/facebook-wall.txt'
    graph_df = pd.read_table(graph_path, sep='\t', header=None)
    graph_df.columns = ['source', 'target', 'time']
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, 'months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    cc_nodes = sorted(nx.connected_components(graph_nx.to_undirected()), key=len, reverse=True)[0] # biggest cc

    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}

    qs = [0.25, 0.5, 1, 2, 4]
    ps = [0.25, 0.5, 1, 2, 4]
    walk_lengths = [32, 64]
    num_walks_list = [3, 10]
    create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name, cc_nodes)
