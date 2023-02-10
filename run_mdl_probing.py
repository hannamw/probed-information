import math
from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import torch 
# from torch.utils.data import DataLoader, ConcatDataset

from sklearn.linear_model import LogisticRegression
# import pytorch_lightning as pl
from transformers import AutoConfig
from probing import get_reps
from dataset import WORDS_IN_SENTENCE

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-cased')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--probe_runs', type=int, default=1)
parser.add_argument('--unmasked', action='store_true')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
# pl_accelerator = "gpu" if device == 'cuda' else device 
# n_devices = torch.cuda.device_count() if pl_accelerator == 'gpu' else 1

model_config = AutoConfig.from_pretrained(args.model_name)
n_layers = model_config.num_hidden_layers + 1
train, _, _ = get_reps(args.model_name, batch_size=args.batch_size)

# considering the smallest dataset for all of the words, we'll 
# take samples of sizes corresponding to percents of that dataset's length
minimum_dataset_size = min(len(train_ds) for train_ds in train)
percents = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
train_ends = [math.floor(0.01 * percent * minimum_dataset_size) for percent in percents]
if args.unmasked:
    train, _, _ = get_reps(args.model_name, batch_size=args.batch_size, masked=False)
    for i, train_ds in enumerate(train):
        assert len(train_ds) >= train_ends[-1], f"Dataset {i} too short: should be {train_ends[-1]}, but is {len(train_ds)}"

# results_array: trials x words x layers x percents
train_accs_array = torch.zeros([args.probe_runs, WORDS_IN_SENTENCE, n_layers, len(percents) - 1])
# entropy (mean) for the next segment, given a probe trained on all prior data
ns_ents_array = torch.zeros([args.probe_runs, WORDS_IN_SENTENCE, n_layers, len(percents) - 1])
# as above, but summed
ns_esums_array = torch.zeros([args.probe_runs, WORDS_IN_SENTENCE, n_layers, len(percents) - 1])

with tqdm(total=args.probe_runs * WORDS_IN_SENTENCE * n_layers * (len(train_ends) - 1)) as pbar:
    for probe_run in range(args.probe_runs):
        for word in range(WORDS_IN_SENTENCE):
            train_dataset = train[word]

            # we're going to sample some examples from the training dataset
            random_indices = torch.randperm(len(train_dataset))

            # but we need to make sure there is at least one example of each label
            panic = 0
            while len(set(train_dataset.labels[random_indices[:train_ends[0]]].tolist())) < 2:
                random_indices = torch.randperm(len(train_dataset))
                if panic >= 100:
                    raise RuntimeError("couldn't find a good set of examples!")  # You messed something up if this happens!
                panic += 1

            # and probe every layer
            for layer in range(n_layers):
                layer_dataset = train_dataset.get_layer_dataset(layer)
                for p, (percent, train_end, next_segment_end) in enumerate(zip(percents, train_ends, train_ends[1:])):
                    # Here, we build a new dataset containing just a subset of the data
                    train_subset_reps, train_subset_labels = layer_dataset[random_indices[:train_end]]

                    # But, it's somewhat unfair that smaller datasets mean that the probes will see 
                    # fewer training examples for a given number of epochs. So, we repeat the training
                    # examples to balance out this inequity.
                    n_repetitions = math.floor(100 / percent)
                    reps_repeated = np.repeat(train_subset_reps.cpu().numpy(), n_repetitions, axis=0)
                    labels_repeated = np.repeat(train_subset_labels.cpu().numpy(), n_repetitions, axis=0)

                    # instantiate and train the probe
                    probe = LogisticRegression(penalty='l2')
                    probe = probe.fit(reps_repeated, labels_repeated)

                    predictions = torch.tensor(probe.predict(train_subset_reps.cpu().numpy()))
                    correct = (predictions == train_subset_labels)

                    train_acc = torch.mean(correct.float())
                    train_accs_array[probe_run, word, layer, p] = train_acc

                    # now, the crucial part: get probe predictions for the next segment
                    ns_reps, ns_labels = layer_dataset[random_indices[train_end:next_segment_end]]

                    ns_probs = torch.tensor(probe.predict_proba(ns_reps.cpu().numpy()))

                    # then compute the entropy of the next segment
                    correct_ns_probs = ns_probs[torch.arange(len(ns_labels)), ns_labels] #torch.where(ns_labels.bool(), ns_probs, 1-ns_probs)
                    ns_entropy = -torch.mean(torch.log2(correct_ns_probs))
                    ns_entropy_sums = -torch.sum(torch.log2(correct_ns_probs))

                    ns_ents_array[probe_run, word, layer, p] = ns_entropy
                    ns_esums_array[probe_run, word, layer, p] = ns_entropy_sums

                    pbar.update(1)

# Computing the actual description length. There are 2 labels, and 4 in the first segment
# Thus, full_cost_of_first_segment should be 4
n_labels = 2
full_cost_of_first_segment = torch.log2(torch.tensor(n_labels)) * train_ends[0]
description_lengths = full_cost_of_first_segment + ns_esums_array.sum(dim=-1)

results_path = Path(f'results/{args.model_name}')
results_path.mkdir(exist_ok=True, parents=True)
if args.unmasked:
    torch.save(ns_esums_array, results_path/'esums_array_unmasked.pt')
    torch.save(train_accs_array, results_path/'mdl_accuracy_unmasked.pt')
    torch.save(description_lengths, results_path/'description_length_unmasked.pt')
else:
    torch.save(ns_esums_array, results_path/'esums_array.pt')
    torch.save(train_accs_array, results_path/'mdl_accuracy.pt')
    torch.save(description_lengths, results_path/'description_length.pt')
