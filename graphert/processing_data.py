import networkx as nx
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import pandas as pd
import numpy as np
from os.path import join, exists
import pickle


class TemporalGraph():
    def __init__(self, data, time_granularity, dataset_name):
        '''
        :param data: DataFrame- source, target, time, weight columns
        :param time_granularity: 'day', 'week', 'month', 'year' or 'hour'
        '''
        data['day'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).day)
        data['week'] = data['time'].apply(
            lambda timestamp: (datetime.utcfromtimestamp(timestamp)).isocalendar()[1])
        data['month'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).month)
        data['year'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).year)
        data['hour'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).hour)
        if 'weight' not in data.columns:
            data['weight'] = 1

        if dataset_name == 'facebook':
            data = data[((data['year'] == 2006) & (data['month'] >= 8)) | (data['year'] > 2006)]

        elif dataset_name == 'enron_m':
            data = data[
                ((data['year'] < 2002) & (data['year'] >= 1999)) | ((data['year'] == 2002) & (data['month'] < 7))]

        self.data = data
        self.time_granularity = time_granularity
        self.time_columns, self.step = self._get_time_columns(time_granularity)
        self.static_graph = self.get_static_graph()
        self.data['time_index'] = self.data.apply(self._get_time, axis=1)

    def get_static_graph(self):

        g = nx.from_pandas_edgelist(self.data, source='source', target='target', edge_attr=['weight'],
                                    create_using=nx.MultiDiGraph())
        self.nodes = g.nodes()
        return g

    def filter_nodes(self, thresh=5):
        nodes2filter = [node for node, degree in self.static_graph.degree() if degree < thresh]
        return nodes2filter

    def get_temporal_graphs(self, min_degree, mode='dynamic'):
        '''

        :param filter_nodes: int.  filter nodes with degree<min_degree in all time steps
        :param mode: if not 'dynamic', add all nodes to the current time step without edges
        :return: dictionary. key- time step, value- nx.Graph
        '''
        G = {}
        for t, time_group in self.data.groupby(self.time_columns):
            time_group = time_group.groupby(['source', 'target'])['weight'].sum().reset_index()
            g = nx.from_pandas_edgelist(time_group, source='source', target='target', edge_attr=['weight'],
                                        create_using=nx.DiGraph())
            if mode != 'dynamic':
                g.add_nodes_from(self.nodes)
            g.remove_nodes_from(self.filter_nodes(min_degree))
            G[self.get_date(t)] = g
        self.graphs = G
        return G

    def _get_time(self, x):
        if 'week' in self.time_columns:
            return datetime.strptime(f"{x.year}-W{x.week}" + '-1', "%Y-W%W-%w")
        elif 'hour' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=x.day, hour=x.hour)
        elif 'day' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=x.day)
        elif 'month' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=1)
        elif 'year' in self.time_columns:
            return datetime(year=x.year, month=1, day=1)

    @staticmethod
    def _time_weight(x, t, time_granularity):
        w = x.weight
        delta_t_in_sec = (t - x['time_index']).total_seconds()
        if time_granularity == 'hours':
            delta_t = delta_t_in_sec / 3600
        elif time_granularity == 'days':
            delta_t = delta_t_in_sec / 3600 / 24
        elif time_granularity == 'weeks':
            delta_t = delta_t_in_sec / 3600 / 24 / 7
        elif time_granularity == 'months':
            delta_t = delta_t_in_sec / 3600 / 24 / 30
        elif time_granularity == 'years':
            delta_t = delta_t_in_sec / 3600 / 24 / 365
        return w * np.log2(1 + 1 / (delta_t + 1))

    def get_date(self, t):
        time_dict = dict(zip(self.time_columns, t if type(t) == tuple else [t]))
        if self.time_granularity == 'hours':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=time_dict['day'],
                            hour=time_dict['hour'])
        elif self.time_granularity == 'days':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=time_dict['day'])
        elif self.time_granularity == 'months':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=1)
        elif self.time_granularity == 'weeks':
            date_year = datetime(year=time_dict['year'], month=1, day=1)
            return date_year + timedelta(days=float((time_dict['week'] - 1) * 7))
        elif self.time_granularity == 'years':
            return datetime(year=time_dict['year'], month=1, day=1)
        else:
            raise Exception("not valid time granularity")

    @staticmethod
    def _get_time_columns(time_granularity):
        if time_granularity == 'hours':
            group_time = ['year', 'month', 'day', 'hour']
            step = timedelta(hours=1)
        elif time_granularity == 'days':
            group_time = ['year', 'month', 'day']
            step = timedelta(days=1)
        elif time_granularity == 'weeks':
            group_time = ['year', 'week']
            step = timedelta(weeks=1)
        elif time_granularity == 'months':
            group_time = ['year', 'month']
            step = relativedelta(months=1)
        elif time_granularity == 'years':
            group_time = ['year']
            step = relativedelta(years=1)
        else:
            raise Exception("not valid time granularity")
        return group_time, step


def load_dataset(graph_df, dataset_name, time_granularity):
    temporal_g = TemporalGraph(data=graph_df, time_granularity=time_granularity, dataset_name=dataset_name)
    graph_df = temporal_g.data
    graph_df['time'] = graph_df['time_index']
    graph_nx = nx.from_pandas_edgelist(graph_df, 'source', 'target', edge_attr=['time'],
                                       create_using=nx.MultiDiGraph())
    return graph_nx, temporal_g


if __name__ == '__main__':
    # load facebook
    dataset_name = 'facebook'
    graph_path = 'data/facebook/facebook-wall.txt'
    graph_df = pd.read_table(graph_path, sep='\t', header=None)
    graph_df.columns = ['source', 'target', 'time']
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, 'months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    print(graphs)

    # load monthly enron
    # dataset_name = 'enron_m'
    # graph_path =  'data/enron/out.enron'
    # graph_df = pd.read_table(graph_path, sep=' ', header=None)
    # graph_df.columns = ['source', 'target', 'weight', 'time']
    # graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
    # graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    #
    # load game of thrones
    # dataset_name = 'game_of_thrones'
    # with open(join('data', 'gameofthrones/gameofthrones_2017_graphs_dynamic.pkl'), 'rb') as f:
    #     graphs = pickle.load(f)
    #
    # #load formula1
    # dataset_name = 'formula'
    # with open(join('data', 'formula/formula_2019_graphs_dynamic.pkl'), 'rb') as f:
    #     graphs = pickle.load(f)
