# GraphERT

This repository provides a reference implementation of *GraphERT* as described in the paper:<br>
> GraphERT-- Transformers-based Temporal Dynamic Graph Embedding <br>
> Moran Beladev, Gilad Katz, Lior Rokach, Uriel Singer, Kira Radinsky.<br>
> CIKM’23 – October 2023, Birmingham, United Kingdom. <br>
> [Link]()


### Basic Usage

### Data ###
All our data is accessible in the "data" folder. 
The `_dynamic` suffix stands for dynamic graphs, having different number of nodes per time step.

The  `_static` suffix stands for static graph, having same number of nodes per time step. 
To achieve that we created all nodes in each time steps, nodes that do not exist at that time step are isolated.

#### Input
The data should include - source node, target node, time of interaction, weight(optional) as csv file (graph_df).
	
	node1_id_int node2_id_int time_timestamp <weight_float, optional>

To create temporal graphs structure as dict, with time as key and nx.Graph as value, use `processing_data` file, 
with examples of creating the data for each dataset- facebook, enron, game of thrones and formula. 

```python
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
```

#### Create random walks

According to the method describing in our paper, each graph time step is converted to a list of sentences using random walks. 

<img src="https://ibb.co/mbGL6SZ" width="600">

To create the random walks, you can follow the `create_random_walks.py` file. You can define the set of parameters used to create the random walks:
- `p` and `q` parameters affect the traverse method in the graph as explained in node2vec.
- `walk_length`(L), each sentence max length
- `num_walks` (gamma)- number of walks starting from each node
- 
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

#### Train the model
To train *GraphERT* you can follow the train_model.py for full flow example.
```python
    dataset_name = 'facebook'
    walk_len = 32
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_3.csv'
    #train a no
    train_graph_tokenizer(random_walk_path, dataset_name, walk_len)
    train_only_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=100_000)
    train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=100_000)
    train_2_steps_model(random_walk_path, dataset_name, walk_len, sample_num=100_000)
   
```



#### output
`model.get_embeddings()` - > numpy array of shape (number of time steps, graph vector dimension size)

	shape = [num_of_time_steps, dim_of_representation]

#### TdGraphEmbed.get_documents_from_graph

According to the method describing in our paper, each graph time step is converted to a list of sentences 
of type `[TaggedDocument(doc, [time])]`. 

<img src="https://i.ibb.co/ZfxYvtB/graph2doc.png" width="600">

You can control the graph to document building process by updating the parameters in the config file: 
- `p` and `q` parameters affect the traverse method in the graph as explained in node2vec.
- `walk_length`(L), each sentence in the document max length
- `num_walks` (gamma)- number of walks starting from each node,
 will affect the number of sentences in the document representing the graph. 

#### Training the model ####

We train our model described in the paper, using the following architecture:
<img src="https://i.ibb.co/Z8g3qt7/g2v.png" width="400"/>

We use doc2vec code in order to apply this architecture.
You can control the doc2vec training parameters by updating the parameters in the config file.


### Citing ###
If you find tdGraphEmbed useful for your research, please consider citing the following paper:

GraphERT-- Transformers-based Temporal Dynamic Graph Embedding
Moran Beladev, Gilad Katz, Lior Rokach, Uriel Singer, Kira Radinsky.<br>
CIKM’23 – October 2023, Birmingham, United Kingdom.
For questions, please contact me at `moranbeladev90@gmail.com`.
