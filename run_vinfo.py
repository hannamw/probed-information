import argparse
from pathlib import Path

import torch 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoConfig

from dataset import WORDS_IN_SENTENCE
from probing import Probe, get_reps, probe_check, train_probes


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-cased')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--probe_runs', type=int, default=10)
parser.add_argument('--unmasked', action='store_true')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
pl_accelerator = "gpu" if device == 'cuda' else device 
n_devices = torch.cuda.device_count() if pl_accelerator == 'gpu' else 1

if not probe_check(args.model_name, args.probe_runs, masked=(not args.unmasked)):
    print("Probes must be trained prior to reflection experiments. Training new probes.")
    train_probes(args.model_name, args.probe_runs, masked=(not args.unmasked))

train, valid, test = get_reps(args.model_name, batch_size=args.batch_size, masked=(not args.unmasked))
probes_dir = 'probes' if not args.unmasked else 'probes_unmasked'

model_config = AutoConfig.from_pretrained(args.model_name)
n_layers = model_config.num_hidden_layers + 1

# results_array: split x trials x words x layers
accuracies = torch.zeros([3, args.probe_runs, WORDS_IN_SENTENCE, n_layers])
entropies = torch.zeros([3, args.probe_runs, WORDS_IN_SENTENCE, n_layers])
v_infos = torch.zeros([3, args.probe_runs, WORDS_IN_SENTENCE, n_layers])
for split, split_dataset in enumerate([train, valid, test]):
    for probe_run in range(args.probe_runs):
        for word in range(WORDS_IN_SENTENCE):
            word_dataset = split_dataset[word]
            n_plural = word_dataset.labels.sum()
            n_singular = len(word_dataset) - n_plural
            baseline_entropy = -(torch.log2(n_plural/len(word_dataset)) + torch.log2(n_singular/len(word_dataset)))/2

            for layer in range(n_layers):
                layer_dataset = word_dataset.get_layer_dataset(layer)
                dataloader = DataLoader(layer_dataset, batch_size=args.batch_size)

                path = Path(f'{probes_dir}/{args.model_name}/run-{probe_run}/word-{word}/layer-{layer}')
                probe = Probe(model_config.hidden_size)
                probe.load_state_dict(torch.load(path/'state_dict.pt'))
                for param in probe.parameters():
                    param.requires_grad = False

                trainer = pl.Trainer(accelerator=pl_accelerator, devices=n_devices, default_root_dir=str(path))
                probs = torch.cat(trainer.predict(probe, dataloader), dim=0).cpu()
                preds = probs > 0.5
                correct = preds == layer_dataset.labels
                accuracies[split, probe_run, word, layer] = correct.float().mean()

                correct_probs = torch.where(layer_dataset.labels.bool(), probs, 1 - probs)
                entropy = -torch.mean(torch.log2(correct_probs))
                v_info = baseline_entropy - entropy

                entropies[split, probe_run, word, layer] = entropy
                v_infos[split, probe_run, word, layer] = v_info

results_path = Path(f'results/{args.model_name}')
results_path.mkdir(exist_ok=True, parents=True)
if args.unmasked:
    torch.save(accuracies, results_path/'accuracy_unmasked.pt')
    torch.save(entropies, results_path/'entropy_unmasked.pt')
    torch.save(v_infos, results_path/'v_info_unmasked.pt')
else:
    torch.save(accuracies, results_path/'accuracy.pt')
    torch.save(entropies, results_path/'entropy.pt')
    torch.save(v_infos, results_path/'v_info.pt')


