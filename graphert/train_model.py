import torch
import pandas as pd
from datasets import Dataset
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer, BertConfig, PreTrainedTokenizerFast, AutoModelForSequenceClassification

import numpy as np
from datasets import load_metric
from tqdm import tqdm

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss

from graphert.train_tokenizer import train_graph_tokenizer


class Temporal_Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


class BertForTemporalClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.temporal_num_labels = config.temporal_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # temporal classification part
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        loss = temporal_loss
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMlmTemporalClassification(BertPreTrainedModel):
    def __init__(self, config, temporal_weight=5):
        super().__init__(config)
        self.temporal_num_labels = config.temporal_num_labels
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.mlm = BertOnlyMLMHead(config)
        self.init_weights()
        self.temporal_weight = temporal_weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # mlm part
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        # temporal classification part
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        loss = masked_lm_loss + self.temporal_weight * temporal_loss
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss, masked_lm_loss, temporal_loss) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def train_mlm(dataset, graph_tokenizer, dataset_name):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=graph_tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="./",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_steps=0,
        save_total_limit=0,
    )

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=64
    )

    model = BertForMaskedLM(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(f'datasets/{dataset_name}/models/masking_model')


def tokenize_function(graph_tokenizer, examples, sent_col_name):
    return graph_tokenizer(examples[sent_col_name], padding='max_length', truncation=True)


def train_2_steps_model(random_walk_path, dataset_name, walk_len, sample_num=None):
    # bert model for classification, based on mlm

    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'datasets/{dataset_name}/models/graph_tokenizer.tokenizer.json', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"
    num_classes = len(set(data_df['time']))

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)

    cols = ['input_ids', 'token_type_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=cols + ['time'])
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])

    train_mlm(dataset, graph_tokenizer, dataset_name)

    model = AutoModelForSequenceClassification.from_pretrained(f'datasets/{dataset_name}/models/masking_model/',
                                                               num_labels=num_classes)
    dataset = dataset.map(lambda examples: {'labels': list(examples['time'].numpy())}, batched=True)
    dataset.set_format(type='torch')

    dataset_test_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_test_dataset['train']
    test_dataset = dataset_test_dataset['test']

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir=f"datasets/{dataset_name}/output",
                                      per_device_train_batch_size=32,
                                      logging_strategy="steps",
                                      num_train_epochs=5,
                                      seed=0, save_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=compute_metrics)

    trainer.train()
    torch.save(model, f'datasets/{dataset_name}/models/time_classification_after_masking')


def train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=None):
    # train mlm and temporal model together (TM + MLM)
    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'datasets/{dataset_name}/models/graph_tokenizer.tokenizer.json', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)
    cols = ['input_ids', 'attention_mask']
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
    dataset.set_format(type='torch', columns=cols + ['time', 'p', 'q'])

    labels = dataset['input_ids']
    mask = dataset['attention_mask']
    temporal_labels = dataset['time']
    num_classes = len(set(dataset['time']))

    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != graph_tokenizer.cls_token_id) * (
            input_ids != graph_tokenizer.pad_token_id) * (input_ids != graph_tokenizer.sep_token_id) * (
                       input_ids != graph_tokenizer.unk_token_id)
    selection = ((mask_arr).nonzero())
    input_ids[selection[:, 0], selection[:, 1]] = graph_tokenizer.mask_token_id

    d = Temporal_Graph_Dataset({'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,
                                'temporal_labels': temporal_labels
                                })
    loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True)

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=walk_len + 4,
        temporal_num_labels=num_classes,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForMlmTemporalClassification(config).to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    epochs = 5

    total_loss = []
    mlm_loss = []
    t_loss = []

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            temporal_labels = batch['temporal_labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                            temporal_labels=temporal_labels)
            # extract loss
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item(), mlm_loss=outputs[1].item(), t_loss=outputs[2].item())

            total_loss.append(loss.item())
            mlm_loss.append(outputs[1].item())
            t_loss.append(outputs[2].item())

            if i % 1000 == 0:
                print(f'loss={np.mean(total_loss)}, mlm_loss={np.mean(mlm_loss)}, t_loss={np.mean(t_loss)}')
                total_loss = []
                mlm_loss = []
                t_loss = []

    torch.save(model, f'datasets/{dataset_name}/models/mlm_and_temporal_model')


def train_only_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=None):
    # train only temporal part (TM)
    data_df = pd.read_csv(random_walk_path, index_col=None)
    graph_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'datasets/{dataset_name}/models/graph_tokenizer.tokenizer.json', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)
    cols = ['input_ids', 'attention_mask']
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
    dataset.set_format(type='torch', columns=cols + ['time', 'p', 'q'])

    num_classes = len(set(dataset['time']))
    temporal_labels = dataset['time']
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=walk_len + 2,
        temporal_num_labels=num_classes,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForTemporalClassification(config).to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    epochs = 5

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            temporal_labels = batch['time'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, temporal_labels=temporal_labels)
            # extract loss
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item(), )
    torch.save(model, f'datasets/{dataset_name}/models/only_temporal')


if __name__ == '__main__':
    dataset_name = 'facebook'
    walk_len = 32
    random_walk_path = f'datasets/{dataset_name}/paths_walk_len_32_num_walks_3.csv'
    train_graph_tokenizer(random_walk_path, dataset_name, walk_len, sample_num=10_000)
    train_only_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=1000)
    train_mlm_temporal_model(random_walk_path, dataset_name, walk_len, sample_num=1000)
    train_2_steps_model(random_walk_path, dataset_name, walk_len, sample_num=1000)
