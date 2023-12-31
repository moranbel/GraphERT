import torch
import sklearn
from datasets import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from graphert.train_model import get_graph_tokenizer, tokenize_function, BertForMlmTemporalClassification, \
    BertForTemporalClassification


def cls_emb(model, examples, t):
    # get per path the cls embedding and the probability for the gt time step
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outputs = model.bert(input_ids=examples['input_ids'].to(device),
                         attention_mask=examples['attention_mask'].to(device))
    embs = outputs.last_hidden_state[:, 0, :]
    pooled_output = outputs[1]
    pooled_output = model.dropout(pooled_output)
    logits = model.classifier(pooled_output)
    probs = torch.nn.functional.softmax(logits, dim=1)[:, t]
    return {'cls_emb': embs.cpu().detach().numpy(), 'probs': probs.cpu().detach().numpy()}


def get_temporal_embeddings(model_path):
    '''
        get temporal embeddings by the last layer (TXd)
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval().to(device)
    return model.classifier.weight.cpu().detach().numpy()


def get_embeddings_by_paths_average(data_df: pd.DataFrame, model_path: str, dataset_name: str, walk_len: int):
    '''
    get the embeddings by averaging the embeddings of all paths in each time
    :param data_df: Pd.DataFrame of the random_walk_path csv
    :param model_path: torch ckpt trained model
    :param dataset_name:
    :param walk_len:
    :return:
    t_emb_mean- using cls embedding of each path and averaging them per time. np.array- TXd
    t_prob- the probability for the true time step per path (using the softmax layer of the classification)
    t_emb_weighted_mean- using cls embedding of each path and average them with a weight by the probability of each time.
    If a path has high probability to be classified with the time step, it means this path better represent the time step,
    that is why we want to use its embedding with the assigned higher probability. np.array- TXd
    '''
    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len)
    t_emb_mean = dict()
    t_emb_weighted_mean = dict()
    t_prob = dict()
    for t, t_df in tqdm(data_df.groupby('time'), total=max(data_df['time'])):
        dataset_t = Dataset.from_pandas(t_df)
        dataset_t = dataset_t.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                                  batch_size=256)
        cols = ['input_ids', 'token_type_ids', 'attention_mask']
        dataset_t.set_format(type='torch', columns=cols + ['time'])
        dataset_t = dataset_t.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])

        dataset_t = dataset_t.map(lambda examples: cls_emb(model, examples, t), batched=True, batch_size=256)
        dataset_t = dataset_t.remove_columns(['input_ids', 'attention_mask'])

        t_emb_mean[t] = dataset_t['cls_emb'].numpy().mean(axis=0)
        t_emb_weighted_mean[t] = dataset_t['probs'].unsqueeze(axis=0).matmul(
            dataset_t['cls_emb']).squeeze().cpu().detach().numpy()
        t_prob[t] = dataset_t['probs'].numpy().mean(axis=0)

    return t_emb_mean, t_emb_weighted_mean, t_prob


if __name__ == '__main__':
    dataset_name = 'facebook'
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer (TXd)
    temporal_embeddings = get_temporal_embeddings(model_path)

    # get temporal embeddings by averaging the paths embeddings per time
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_3.csv'
    data_df = pd.read_csv(random_walk_path, index_col=None)
    t_emb_mean, t_emb_weighted_mean, t_prob = get_embeddings_by_paths_average(data_df, model_path, dataset_name,
                                                                              walk_len=32)
    t_emb_mean, t_emb_weighted_mean = np.array(list(t_emb_mean.values())), np.array(list(t_emb_weighted_mean.values()))

    # plot similarity
    print("Temporal similarity plot:")
    similarity = sklearn.metrics.pairwise.cosine_similarity(temporal_embeddings, temporal_embeddings)
    print(sns.heatmap(similarity, vmin=0, vmax=1))
