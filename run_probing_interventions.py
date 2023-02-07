from pathlib import Path
from functools import partial
import argparse

from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from transformers import AutoModelForMaskedLM, AutoTokenizer

from dataset import load_intervention_dataset, collate_fn, get_start_ends
from probing import probe_check, train_probes, load_probe_params
from utils import reflect, get_word_number_mapping, get_module_by_layer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='distilroberta-base')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--probe_runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=40)
args = parser.parse_args()

supported_models = {'bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'distilbert-base-cased', 'distilroberta-base'}
assert args.model_name in supported_models, f'Requested model {args.model_name} is not in the set of supported models: {supported_models}.'

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
model.to(device)
n_layers = model.config.num_hidden_layers + 1  # the + 1 is for the embedding layer
model.eval()

word_number_mapping = get_word_number_mapping(tokenizer)
_, _, dataset = load_intervention_dataset(tokenizer, masked=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

# there are 7 different words to intervene on in each sentence: 
# ART1, ADJ, SUBJ, ADV, VERB, ART2, OBJ
WORDS_TO_INTERVENE = 7
ART1, ADJ, SUBJ, ADV, VERB, ART2, OBJ = list(range(7))

def run_with_hook(model, inp, layer, hook_fn):
    """
    Inspired by Neel Nanda's TransformerLens, which doesn't yet support MLMs
    Inserts the hook at the given layer, runs the model, and removes the hook
    Only does very simple hooking (i.e. transformer block output)
    """
    module_to_hook = get_module_by_layer(model, args.model_name, layer)
    hook_handle = module_to_hook.register_forward_hook(hook_fn)
    with torch.inference_mode():
        logits = model(**inp)['logits']
    hook_handle.remove()
    return logits

def interchange_intervention_hook(module, input,
    output: TT["batch", "pos", "d_model"],
    cache: torch.Tensor,
    idxs: torch.Tensor,
    alternate_idxs: torch.Tensor, 
    layer: int
) -> TT["batch", "pos", "d_model"]:
    if layer != 0:
        output, = output
    for i, ((s,e), (a_s, a_e)) in enumerate(zip(idxs, alternate_idxs)):
        output[i, s:e, :] = cache[i, a_s:a_e, :]
    return output if layer == 0 else (output,)

if True:
    # Perform the interchange intervention experiments only
    # The last dim stores the sum of prob assigned to verbs that agree [0] and disagree [1]
    interchange_results = torch.zeros([len(dataset), WORDS_TO_INTERVENE, n_layers, 2])
    batch_start = 0
    for batch in tqdm(dataloader):
        batch_len = len(batch[0])  # len(batch) is 3, since it's a 3-tuple. could be < args.batch_size
        batch_end = batch_start + batch_len
        sentences, alternate_sentences, labels = batch

        tokens = tokenizer(sentences, return_tensors='pt', padding='longest').to(device)
        alternate_tokens = tokenizer(alternate_sentences, return_tensors='pt', padding='longest').to(device)
        alternate_hidden_states = model(**alternate_tokens, output_hidden_states=True)['hidden_states']
        start_ends = get_start_ends(tokens)
        alternate_start_ends = get_start_ends(alternate_tokens)

        for word in range(WORDS_TO_INTERVENE):
            for layer in range(n_layers):
                temp_hook_fn = partial(interchange_intervention_hook, 
                                    cache=alternate_hidden_states[layer], 
                                    idxs=start_ends[:, word], 
                                    alternate_idxs=alternate_start_ends[:, word],
                                    layer=layer)

                # run the model and get only those logits corresponding to the masked verb
                interchange_logits = run_with_hook(model, tokens, layer, temp_hook_fn)[torch.arange(batch_len), start_ends[:, VERB, 0]].cpu()
                interchange_probs = F.softmax(interchange_logits, dim=-1)

                # j[0] is singular probability, j[1] is plural
                probs_by_number = torch.einsum('bi,ij->bj', interchange_probs, word_number_mapping)
                correct_indices = torch.stack([labels, 1-labels], dim=1)
                # j[0] is agree probability, j[1] is disagree
                probs_by_correctness = torch.gather(probs_by_number[:,:2], -1, correct_indices)
                interchange_results[batch_start:batch_end, word, layer] = probs_by_correctness
                torch.cuda.empty_cache()

        batch_start = batch_end

    results_path = Path(f'results/{args.model_name}')
    results_path.mkdir(exist_ok=True, parents=True)
    torch.save(interchange_results, results_path/'interchange.pt')

# Perform the reflection intervention experiments only 
if not probe_check(args.model_name, args.probe_runs):
    print("Probes must be trained prior to reflection experiments. Training new probes.")
    train_probes(args.model_name, args.probe_runs, epochs=args.epochs)


def reflection_intervention_hook(module, input,
    output: TT["batch", "pos", "d_model"],
    hyperplane: torch.Tensor,
    offset: torch.Tensor,
    idxs: torch.Tensor,
    layer:int
) -> TT["batch", "pos", "d_model"]:
    if layer != 0:
        output, = output
    for i, (s, e) in enumerate(idxs):
        output[i, s:e, :] = reflect(output[i, s:e, :], hyperplane, offset)
    return output if layer == 0 else (output,)

reflection_results = torch.zeros([args.probe_runs, len(dataset), WORDS_TO_INTERVENE, n_layers, 2])
for probe_run in range(args.probe_runs):
    print(f"starting run {probe_run+1}/{args.probe_runs}")
    probe_params = load_probe_params(args.model_name, probe_run, device=device)
    batch_start = 0
    for batch in tqdm(dataloader):
        batch_len = len(batch[0])
        batch_end = batch_start + batch_len
        sentences, _, labels = batch

        tokens = tokenizer(sentences, return_tensors='pt', padding='longest').to(device)
        start_ends = get_start_ends(tokens)

        for word in range(WORDS_TO_INTERVENE):
            for layer in range(n_layers):
                temp_hook_fn = partial(reflection_intervention_hook, 
                                  hyperplane=probe_params[word, layer, :model.config.hidden_size], 
                                  offset=probe_params[word, layer, model.config.hidden_size],
                                  idxs=start_ends[:, word],
                                  layer=layer)

                reflection_logits = run_with_hook(model, tokens, layer, temp_hook_fn)[torch.arange(batch_len), start_ends[:, VERB, 0]].cpu()
                reflection_probs = F.softmax(reflection_logits, dim=-1)

                # j[0] is singular probability, j[1] is plural
                probs_by_number = torch.einsum('bi,ij->bj', reflection_probs, word_number_mapping)
                correct_indices = torch.stack([labels, 1-labels], dim=1)
                # j[0] is agree probability, j[1] is disagree
                probs_by_correctness = torch.gather(probs_by_number[:,:2], -1, correct_indices)
                reflection_results[probe_run, batch_start:batch_end, word, layer] = probs_by_correctness 

        batch_start = batch_end

results_path = Path(f'results/{args.model_name}')
results_path.mkdir(exist_ok=True, parents=True)
torch.save(reflection_results, results_path/'reflection.pt')
