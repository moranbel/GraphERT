from scipy import spatial
from scipy.stats import spearmanr
import datetime
from datetime import datetime as dt
import numpy as np
import pandas as pd

from graphert.temporal_embeddings import get_temporal_embeddings
from graphert.train_model import BertForMlmTemporalClassification, BertForTemporalClassification


def evaluate_anomalies(embs_vectors: np.array, times: list, anoms: list, google_df: pd.DataFrame = None):
    '''

    :param embs_vectors: temporal graph vectors for each time step. numpy array of shape (number of timesteps, graph vector dimension size)
    :param times: list of datetime of all graph's times
    :param anoms: list of anomalies times
    :param google: google trend data in case we have, with 'Day' and 'google' columns to represent the volume per day.
    :return:
    '''
    measures_df = pd.DataFrame(columns=['K', 'Recall', 'Precision'])
    ks = [5, 10]
    dist = np.array([spatial.distance.cosine(embs_vectors[i + 1], embs_vectors[i])
                     for i in range(0, len(embs_vectors) - 1)])
    for k in ks:
        top_k = (-dist).argsort()[:k]
        top_k = np.array(times)[top_k]
        tp = np.sum([1 if anom in top_k else 0 for anom in anoms])
        recall_val = tp / len(anoms)
        precision_val = tp / k
        measures_df = measures_df.append({'K': k, 'Recall': recall_val, 'Precision': precision_val},
                                         ignore_index=True)
    if google_df is not None:
        corr, pval = spearmanr(dist, google_df['google'].values)
        print(f'Spearman correlation: {corr}, p-value: {pval}')
    print(measures_df)


if __name__ == "__main__":
    dataset_name = 'game_of_thrones'
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)

    google_trends_df = pd.read_csv(f"data/{dataset_name}/google_trends.csv",
                                   parse_dates=['Day'], date_parser=
                                   lambda x: dt.strptime(x, "%d-%m-%y"))

    anoms = [datetime.date(2017, 7, 17), datetime.date(2017, 7, 24),
             datetime.date(2017, 7, 31),
             datetime.date(2017, 8, 7), datetime.date(2017, 8, 14), datetime.date(2017, 8, 21),
             datetime.date(2017, 8, 28)]
    times = google_trends_df['Day'].apply(lambda x: x.date()).values
    evaluate_anomalies(temporal_embeddings, times=times, anoms=anoms,
                       google_df=google_trends_df)
