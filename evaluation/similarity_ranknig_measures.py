import numpy as np
import pandas as pd

from evaluation.graph_similarity import graph2graph_similarity, graph2graph_mcs
from graphert.processing_data import load_dataset
from graphert.temporal_embeddings import get_temporal_embeddings


def precision_at_k(predicted, real, k):
    precision = []
    for t, graph in predicted.iterrows():
        top_predicted = graph.sort_values().index[1:k + 1]
        top_real = real.loc[t].sort_values(ascending=False).index[1:k + 1]
        precision.append(len(set(top_predicted).intersection(set(top_real))) / k)
    return np.mean(precision)


def MAP_at_k(predicted, real, k):
    average_precision = []
    for t, graph in predicted.iterrows():
        top_predicted = graph.sort_values(ascending=False).index[1:k + 1]
        top_real = real.loc[t].sort_values(ascending=False).index[1:k + 1]
        cur_average_precision = 0
        hits = 1
        for i, g_index in enumerate(top_predicted):
            if g_index in top_real:
                cur_average_precision += hits / (i + 1)
                hits += 1
        average_precision.append(cur_average_precision / hits)
    return average_precision


def MRR(predicted, real):
    mrr = []
    for t, graph in predicted.iterrows():
        curr_real_t = real.loc[t]
        curr_real_t = curr_real_t.drop(t)
        top1_real = curr_real_t.sort_values(ascending=False).index[0]
        top1_predicted_rank = list(graph.sort_values(ascending=False).index).index(top1_real)
        if top1_predicted_rank == 0:
            top1_predicted_rank = 1
        mrr.append(1 / top1_predicted_rank)
    return mrr


def eval_similarity(graphs, graphs_embs, similarity_matrix_gt):
    graphs_dict = {}
    res_df = pd.DataFrame(columns=['metric', 'value'])
    for time, graph in enumerate(graphs_embs):
        t = list(graphs.keys())[time]
        graphs_dict[t] = graph

    predicted = graph2graph_similarity(graphs_dict)

    for k in [5, 10]:
        res_df = res_df.append(pd.DataFrame({'metric': f"Precision@{k}",
                                             'value': precision_at_k(predicted, similarity_matrix_gt, k)}))
    res_df = res_df.append(pd.DataFrame({'metric': "map",
                                         'value': MAP_at_k(predicted, similarity_matrix_gt, k=10)}))
    res_df = res_df.append(pd.DataFrame({'metric': "mrr",
                                         'value': MRR(predicted, similarity_matrix_gt)}))
    return res_df


if __name__ == "__main__":
    dataset_name = 'facebook'
    graph_path = 'data/facebook/facebook-wall.txt'
    graph_df = pd.read_table(graph_path, sep='\t', header=None)
    graph_df.columns = ['source', 'target', 'time']
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, 'months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)

    similarity_matrix_gt = graph2graph_mcs(graphs)
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)

    eval_similarity(graphs, temporal_embeddings, similarity_matrix_gt)
