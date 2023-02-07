from typing import List

import pandas as pd

import torch
from tqdm import tqdm

from transformers import PreTrainedTokenizer, BatchEncoding
from torch.utils.data import Dataset

WORDS_IN_SENTENCE = 8

class InterventionDataset(Dataset):
    words_in_sentence: int = WORDS_IN_SENTENCE
    sentences: List[str]
    alternate_sentences: List[str]
    labels: torch.Tensor

    def __init__(self, sentences: List[str], alternate_sentences: List[str], labels: torch.Tensor):
        assert len(sentences) == len(alternate_sentences) == len(labels), f'inputs have different lengths: {len(sentences), len(alternate_sentences), len(labels)}'
        assert len(sentences) > 0, f'inputs have length 0'
        self.sentences = sentences
        self.alternate_sentences = alternate_sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (self.sentences[idx], self.alternate_sentences[idx], self.labels[idx])

    def filter(self, idx):
        return InterventionDataset([sent for b, sent in zip(idx, self.sentences) if b], [sent for b, sent in zip(idx, self.alternate_sentences) if b], self.labels[idx])

def collate_fn(exs):
    sentences, alternate_sentences, labels = zip(*exs)
    return list(sentences), list(alternate_sentences), torch.stack(labels)

def add_nowadays(s: str):
    return s[:-1] + ' nowadays.'

def get_masked_sentences(df: pd.DataFrame, mask_token: str):
    sentences = [s.replace(f' {V} ', f' {mask_token} ') for s, V in zip(df['sentence'], df['V'])]
    alternate_sentences = [s.replace(f' {V} ', f' {mask_token} ') for s, V in zip(df['sentence_opposite'], df['V_opposite'])]
    return sentences, alternate_sentences

def get_start_ends(tokenized_batch):
    batch_len = len(tokenized_batch['input_ids'])
    start_ends = torch.zeros([batch_len, WORDS_IN_SENTENCE, 2], dtype=torch.int64)
    for i in range(batch_len):
        for j in range(WORDS_IN_SENTENCE):
            s, e = tokenized_batch.word_to_tokens(i,j)
            start_ends[i, j, 0] = s
            start_ends[i, j, 1] = e
    return start_ends

def load_intervention_dataset(tokenizer: PreTrainedTokenizer, masked: bool=True):
    df = pd.read_csv('this_dataset.csv')

    if masked:
        sentences, alternate_sentences = get_masked_sentences(df, tokenizer.mask_token)
    else:
        sentences, alternate_sentences = df['sentence'], df['sentence_opposite']

    sentences = [add_nowadays(s) for s in sentences]
    alternate_sentences = [add_nowadays(s) for s in alternate_sentences]

    ds = InterventionDataset(sentences, alternate_sentences, torch.tensor(df['label']))
    if masked:
        # in the masked case, where we actually do interventions, we'll filter out examples where interventions can't be done
        tokens = tokenizer(sentences, padding='longest')
        alternate_tokens = tokenizer(alternate_sentences, padding='longest')
        
        start_ends = get_start_ends(tokens)
        alternate_start_ends = get_start_ends(alternate_tokens)
        word_lengths = start_ends[:, :, 1] - start_ends[:, :, 0]
        alternate_word_lengths = alternate_start_ends[:, :, 1] - alternate_start_ends[:, :, 0]
        same_length = torch.all(word_lengths == alternate_word_lengths, dim=-1)
        ds = ds.filter(same_length)

    train, valid, test = InterventionDataset(*ds[:4000]), InterventionDataset(*ds[4000:5000]), InterventionDataset(*ds[5000:6000])
    return train, valid, test

