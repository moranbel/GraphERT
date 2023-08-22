from pathlib import Path
from datasets import Dataset
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer
import pandas as pd
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel


def train_graph_tokenizer(random_walks_file_path, dataset_name, walk_len, sample_num=None):
    # create files for tokenizer training
    data_df = pd.read_csv(random_walks_file_path)
    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)

    text_data = []
    file_count = 0
    for sample in tqdm(dataset):
        sample = sample['sent'].replace('\n', '')
        text_data.append(sample)
        if len(text_data) == 100000:
            # once we get the 100K mark, save to file
            with open(f'datasets/{dataset_name}/graph_paths/chunk_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    with open(f'datasets/{dataset_name}/graph_paths/chunk_{file_count}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))

    # train word level tokenizer
    paths = [str(x) for x in Path(f'datasets/{dataset_name}/graph_paths').glob('**/*.txt')]

    unk_token = '[UNK]'
    tokenizer = Tokenizer(WordLevel({unk_token: 0}, unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=paths, trainer=trainer)
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    )
    tokenizer.enable_truncation(max_length=walk_len)
    tokenizer.save(f'datasets/{dataset_name}/models/graph_tokenizer.tokenizer.json')


if __name__ == '__main__':
    walk_len = 32
    dataset_name = 'facebook'
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_10.csv'
    train_graph_tokenizer(random_walk_path, dataset_name, walk_len, sample_num=10_000)

    # load and test tokenizer
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'datasets/{dataset_name}/models/graph_tokenizer.tokenizer.json', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"
    tokens = graph_tokenizer('90 11')
    print(tokens)
