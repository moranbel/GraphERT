# GraphERT

This repository provides a reference implementation of *GraphERT* as described in the paper:<br>
> GraphERT-- Transformers-based Temporal Dynamic Graph Embedding <br>
> Moran Beladev, Gilad Katz, Lior Rokach, Uriel Singer, Kira Radinsky.<br>
> CIKM’23 – October 2023, Birmingham, United Kingdom. <br>
> [Link]()

<img src="https://i.ibb.co/MZsYM9w/GraphERT.png" width="1000">

### Data ###
All our data is accessible in the "data" folder. 
The `_dynamic` suffix stands for dynamic graphs, having different number of nodes per time step.

The  `_static` suffix stands for static graph, having same number of nodes per time step. 
To achieve that we created all nodes in each time steps, nodes that do not exist at that time step are isolated.

#### Input
The data should include - source node, target node, time of interaction, weight(optional) as csv file (graph_df).
	
	node1_id_int node2_id_int time_timestamp <weight_float, optional>

To create temporal graphs structure as dict, with time as key and nx.Graph as value, use `processing_data.py` file, 
with examples of creating the data for each dataset- facebook, enron, game of thrones and formula. 

```python
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
```

### Create random walks ###

According to the method describing in our paper, each graph time step is converted to a list of sentences using random walks.

To create the random walks, you can follow the `create_random_walks.py` file. You can define the set of parameters used to create the random walks:
- `p` and `q` parameters affect the traverse method in the graph as explained in node2vec.
- `walk_length`(L), each sentence max length
- `num_walks` (gamma)- number of walks starting from each node

```python
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, 'months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    cc_nodes = sorted(nx.connected_components(graph_nx.to_undirected()), key=len, reverse=True)[0] # biggest cc
    graphs = {i: v for i, (k, v) in enumerate(graphs.items())}
    qs = [0.25, 0.5, 1, 2, 4]
    ps = [0.25, 0.5, 1, 2, 4]
    walk_lengths = [32, 64]
    num_walks_list = [3, 10]
    create_random_walks(graphs, ps, qs, walk_lengths, num_walks_list, dataset_name, cc_nodes)
```

### Train the model ###
To train *GraphERT* you can follow the train_model.py for full flow example.
```python
    dataset_name = 'facebook'
    walk_len = 32
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_3.csv'
    #train a node-level tokenizer
    train_graph_tokenizer(random_walk_path, dataset_name, walk_len)
    train_only_temporal_model(random_walk_path, dataset_name, walk_len)
    train_mlm_temporal_model(random_walk_path, dataset_name, walk_len)
    train_2_steps_model(random_walk_path, dataset_name, walk_len)
   
```
First, train a node-level tokenizer (`train_graph_tokenizer`), then you can choose training approach:
- `train_only_temporal_model` (TM model)- train the model only with the temporal loss
- `train_mlm_temporal_model` (TM + MLM model)- train the model the combined masking loss (MLM) and temporal loss (TM)
- `train_2_steps_model`- train the model separately with 2 steps: first, train it with MLM loss, then use the trained model to fine-tune it on the temporal task. 

From the experiments we recommend to use the TM + MLM model.

### Get the temporal embeddings ###
To get the temporal embeddings in

	shape = [num_of_time_steps, dim_of_representation]

use the `temporal_embeddings.py` file, define the trained model path:
```python
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)
```

### Evaluation and visualizations ###
To evaluate the trained temporal graph embeddings on graphs similarity and anomaly detection tasks follow `similarity_ranknig_measures.py` and `anomaly_evaluation.py`.
Use `visualizations.py` to create temporal similarity matrix:

<img src="https://i.ibb.co/xmqDmZR/facebook-g2g.png" width="400">

and trend analysis:

<img src="https://i.ibb.co/GQKRVjc/Screenshot-2023-08-26-at-12-43-16.png" width="400">


### Citing ###
If you find tdGraphEmbed useful for your research, please consider citing the following paper:

GraphERT-- Transformers-based Temporal Dynamic Graph Embedding
Moran Beladev, Gilad Katz, Lior Rokach, Uriel Singer, Kira Radinsky.
CIKM’23 – October 2023, Birmingham, United Kingdom.
For questions, please contact me at `moranbeladev90@gmail.com`.
