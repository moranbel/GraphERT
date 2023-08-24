from scipy import spatial
from scipy.stats import spearmanr
import holoviews as hv
import numpy as np
import pandas as pd
import matplotlib

from graphert.temporal_embeddings import get_temporal_embeddings

matplotlib.use('tkagg')
import datetime

hv.extension('bokeh')


def evaluate_anomalies(embs_vectors, days, anoms, google = None):
    '''

    :param embs_vectors: temporal graph vectors for each time step. numpy array of shape (number of timesteps, graph vector dimension size)
    :param days: list of datetime of all graph's times
    :param anoms: list of anomalies times
    :param google: google trend data in case we have
    :return:
    '''
    measures_df = pd.DataFrame(columns = ['K', 'Recall', 'Precision'])
    ks = [5, 10]
    dist = np.array([spatial.distance.cosine(embs_vectors[i + 1], embs_vectors[i])
                     for i in range(1, len(embs_vectors) - 1)])
    for k in ks:
        top_k = (-dist).argsort()[:k] + 1
        top_k = np.array(days)[top_k]
        tp = np.sum([1 if anom in top_k else 0 for anom in anoms])
        recall_val = tp / len(anoms)
        precision_val = tp / k
        measures_df = measures_df.append({'K': k, 'Recall': recall_val, 'Precision': precision_val},
                                         ignore_index = True)
    if google:
        corr, pval = spearmanr(dist, google.squeeze()[:-1])
        print(f'{corr}, {pval}')
    print(measures_df)
    #todo: add mrr



if __name__ == "__main__":
    dataset_name = 'game_of_thrones'
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)

    anoms = [datetime.date(2017, 7, 17), datetime.date(2017, 7, 24),
             datetime.date(2017, 7, 31),
             datetime.date(2017, 8, 7), datetime.date(2017, 8, 14), datetime.date(2017, 8, 21),
             datetime.date(2017, 8, 28)]
    evaluate_anomalies(temporal_embeddings, days = days, anoms = anoms)