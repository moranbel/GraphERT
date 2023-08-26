import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from scipy import spatial
from datetime import datetime as dt
import pandas as pd
import datetime
import holoviews as hv
import matplotlib.pyplot as plt
from bokeh.plotting import show

from graphert.temporal_embeddings import get_temporal_embeddings, get_embeddings_by_paths_average
from graphert.train_model import BertForMlmTemporalClassification, BertForTemporalClassification

hv.extension('bokeh')


def similarity_heatmap(embs: np.array):
    print("Temporal similarity plot:")
    similarity = sklearn.metrics.pairwise.cosine_similarity(embs, embs)
    return sns.heatmap(similarity, vmin=0, vmax=1)


def plot_trend_over_time(embs: np.array, google_trends_df: pd.DataFrame, t_prob: list, anomalies: list):
    dist = np.array([spatial.distance.cosine(embs[i - 1], embs[i]) for i in range(1, len(embs))])
    scaler = MinMaxScaler()
    dist = scaler.fit_transform(dist.reshape(1, -1).T).squeeze()

    google_trends_df['google'] = scaler.fit_transform(google_trends_df['google'].values.reshape(-1, 1))
    google_trends_df['graphert_prob'] = scaler.fit_transform(np.array(t_prob[1:]).reshape(-1, 1))
    google_trends_df['graphert_delta'] = dist

    googletrends = hv.Curve((google_trends_df['Day'], google_trends_df['google']), 'Time', 'Score',
                            label='Google Trends', line_dash='dashed')
    graphert_delta = hv.Curve((google_trends_df['Day'], google_trends_df['graphert_delta']), 'Time', 'Score',
                              color='gray', label='GraphERT delta_t')
    graphert_prob = hv.Curve((google_trends_df['Day'], google_trends_df['graphert_prob']), 'Time', 'delta_t',
                             color='blue', label='GraphERT probability_t')
    scat_anoms = hv.Scatter(data=google_trends_df[google_trends_df['Day'].isin(anomalies)], xdims='graphert_delta',
                            vdims=['graphert_delta']).opts(
        width=700, height=500, size=10, fill_color="red", fontsize={'labels': 24, 'xticks': 16, 'yticks': 16})
    googletrends.opts(color='gray', line_dash='dashed')
    graphert_delta.opts(color='purple')
    overlay = (graphert_delta * googletrends * graphert_prob * scat_anoms)
    overlay.opts(legend_position='bottom_right')
    return overlay


if __name__ == '__main__':
    dataset_name = 'game_of_thrones'
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)

    # get temporal embeddings by averaging the paths embeddings per time
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_3.csv'
    data_df = pd.read_csv(random_walk_path, index_col=None)
    t_emb_mean, t_emb_weighted_mean, t_prob = get_embeddings_by_paths_average(data_df.sample(10_000), model_path,
                                                                              dataset_name, walk_len=32)

    google_trends_df = pd.read_csv(f"data/{dataset_name}/google_trends.csv",
                                   parse_dates=['Day'], date_parser=
                                   lambda x: dt.strptime(x, "%d-%m-%y"))

    anoms = [datetime.date(2017, 7, 17), datetime.date(2017, 7, 24),
             datetime.date(2017, 7, 31),
             datetime.date(2017, 8, 7), datetime.date(2017, 8, 14), datetime.date(2017, 8, 21),
             datetime.date(2017, 8, 28)]

    heatmap = similarity_heatmap(temporal_embeddings)
    plt.show()
    trend_plot = plot_trend_over_time(embs=temporal_embeddings,
                                      google_trends_df=google_trends_df,
                                      t_prob=list(t_prob.values()), anomalies=anoms)
    show(hv.render(trend_plot))
