import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

import nodebox_linguistics_extended as nle


def eval_word(word: str):
    inf = nle.verb.infinitive(word)
    if inf == "":
        return [0, 0, 0, 1]
    elif nle.verb.present(inf, '3') == word:
        return [1, 0, 0, 0]
    elif nle.verb.present(inf, '*') == word:
        return [0, 1, 0, 0]
    else:
        return [0, 0, 1, 0]

def get_word_number_mapping(tokenizer: PreTrainedTokenizer):
    category_mapping = []
    for i in range(len(tokenizer)):
        token = tokenizer._convert_id_to_token(i)
        if token[0] == 'Ġ':
            token = token[1:]
        category_mapping.append(eval_word(token))
    category_mapping = torch.tensor(category_mapping, dtype=torch.float32)
    return category_mapping


def reflect(vector: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor):
    orthogonal_projection = project(vector, normal, offset=offset)
    reflection = vector - 2 * orthogonal_projection
    return reflection


def project(vector: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor):
    # vector: word_length (tokens) x hidden_dim
    # normal: hidden_dim
    # offset: hidden_dim
    numer = torch.einsum('ik,k->i', vector, normal)
    denom = torch.einsum('k,k->', normal, normal)
    numer += offset
    scaling_factor = numer / denom
    projection = torch.einsum('i,k->ik', scaling_factor, normal)
    return projection

def get_module_by_layer(model: PreTrainedModel, model_name: str, layer: int):
    # Here, layer == 0 denotes the embeddings; the rest are 1-indexed
    assert layer >= 0, f"negative layers not allowed: {layer}"

    if model_name in {'bert-base-cased', 'bert-large-cased'}:
        base_model = model.bert
        encoder = base_model.encoder
    elif model_name in {'roberta-base', 'roberta-large'}:
        base_model = model.roberta
        encoder = base_model.encoder
    elif model_name == 'distilbert-base-cased':
        base_model = model.distilbert
        encoder = base_model.transformer  # this one's different
    elif model_name == 'distilroberta-base':
        base_model = model.roberta
        encoder = base_model.encoder
    else: 
        raise ValueError(f'Got invalid model name {model_name}')
    
    if layer == 0:
        return base_model.embeddings
    else: 
        layer -= 1
        dummy_child, = encoder.children()
        encoder_modules = list(dummy_child.children())
        assert layer < len(encoder_modules), f"layer too large for model size ({len(encoder_modules)}): {layer}"
        return encoder_modules[layer]