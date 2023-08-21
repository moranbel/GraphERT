from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx

from graphert.processing_data import load_dataset


def create_random_walks(graph_nx: nx.Graph, graphs: dict, qs: list, walk_lengths: list, num_walks_list: list,
                        dataset_name: str):
    cc_nodes = sorted(nx.connected_components(graph_nx.to_undirected()), key=len, reverse=True)[0]  # biggest cc
    for q in qs:
        for walk_len in walk_lengths:
            for num_walks in num_walks_list:
                print(f"q={q}, walk_len={walk_len}, num_walks={num_walks}")
                file_path = f'datasets/{dataset_name}/paths_q{q}_w_{walk_len}_num_walk_{num_walks}.csv'
                data_df_list = []
                nodes = set()
                for i, (time, graph) in enumerate(list(graphs.items())):
                    print(time)
                    graph.remove_nodes_from(
                        [node for node in graph if node not in cc_nodes])  # remove nodes not in the biggest cc
                    graph = graph.to_undirected()
                    nodes = nodes.union(graph.nodes())
                    n2v_model = Node2Vec(graph, num_walks=num_walks, walk_length=walk_len, q=q, workers=4, quiet=True)
                    sents = [" ".join(sent) for sent in n2v_model.walks]
                    data_df_list.append(
                        pd.DataFrame(np.array([sents, [time] * len(sents)]).T, columns=['sent', 'time']))

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

    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}

    qs = [0.5, 1, 0.25]
    walk_lengths = [32, 64, 16]
    num_walks_list = [10, 20]

    create_random_walks(graph_nx, graphs, qs, walk_lengths, num_walks_list, dataset_name)
