from pathlib import Path

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from dataset import load_intervention_dataset, collate_fn, get_start_ends, WORDS_IN_SENTENCE

device = "cuda" if torch.cuda.is_available() else "cpu"
pl_accelerator = "gpu" if device == 'cuda' else device 
n_devices = torch.cuda.device_count() if pl_accelerator == 'gpu' else 1

class RepDataset(Dataset):
    reps: torch.Tensor
    labels: torch.Tensor

    def __init__(self, reps, labels):
        assert reps.size(0) == len(labels), f"Reps and labels have unequal lengths ({reps.size(0)}) vs. ({len(labels)})."
        self.reps = reps  # batch layers hidden_dim
        self.labels = labels 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.reps[idx], self.labels[idx])

    def get_layer_dataset(self, layer:int):
        assert self.reps.ndim == 3, f'Dataset should have 3 dims to select a layer, but has {self.reps.ndim}'
        assert layer >= 0
        return RepDataset(self.reps[:, layer], self.labels)

def collate_fn_reps(exs):
    reps, labels = (torch.stack(x) for x in zip(*exs))
    return reps, labels

def get_reps(model_name:str, batch_size:int=8, masked:bool=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    train, valid, test = load_intervention_dataset(tokenizer, masked=masked)
    rep_datasets = []
    for dataset in [train, valid, test]:
        reps = [[] for _ in range(WORDS_IN_SENTENCE)]
        rep_labels = [[] for _ in range(WORDS_IN_SENTENCE)]
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        for batch in dataloader:
            with torch.inference_mode():
                sentences, _, labels = batch

                tokens = tokenizer(sentences, return_tensors='pt', padding='longest').to(device)
                output = model(**tokens, output_hidden_states=True)
                hidden_states = torch.stack(output['hidden_states'])
                hidden_states = rearrange(hidden_states, 'layers batch position hidden_dim -> batch position layers hidden_dim')

                start_ends = get_start_ends(tokens)
                for word in range(WORDS_IN_SENTENCE):
                    word_reps = []
                    word_labels = []
                    for i in range(len(sentences)):
                        s, e = start_ends[i, word]
                        word_reps.append(hidden_states[i, s:e])
                        word_labels += [labels[i]] * (e - s)
                    word_reps = torch.cat(word_reps, dim=0)
                    reps[word].append(word_reps)
                    rep_labels[word] += word_labels
        rep_dataset = [RepDataset(torch.cat(rs, dim=0), torch.tensor(labels)) for rs, labels in zip(reps, rep_labels)]
        rep_datasets.append(rep_dataset)
    return rep_datasets

class Probe(pl.LightningModule):
    def __init__(self, hidden_size:int):
        super().__init__()

        self.hidden_size = hidden_size
        self.probe = nn.Linear(self.hidden_size, 1)

        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.step_type_accuracy_map = {
            'train': self.train_accuracy,
            'valid': self.valid_accuracy,
            'test': self.test_accuracy
        }

    def forward(self, x):
        y_hat_logits = self.probe(x)
        return y_hat_logits.squeeze(-1)

    def generic_step(self, batch, step_type):
        X, y = batch
        y_hat_logits = self.forward(X)
        loss = F.binary_cross_entropy_with_logits(y_hat_logits, y.float())
        self.log(f'{step_type}_loss', loss)
        pred = torch.round(F.sigmoid(y_hat_logits))

        accuracy_metric = self.step_type_accuracy_map[step_type] 
        accuracy_metric(pred, y)
        self.log(f'{step_type}_acc', accuracy_metric)
        return loss

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.generic_step(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        X, y = batch
        y_hat_logits = self(X)
        return torch.sigmoid(y_hat_logits)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_probes(model_name:str, probe_runs:int, batch_size:int=8, epochs:int=40):
    train, valid, _ = get_reps(model_name, batch_size=batch_size)
    model_config = AutoConfig.from_pretrained(model_name)
    n_layers = model_config.num_hidden_layers + 1

    for probe_run in range(probe_runs):
        for word in range(WORDS_IN_SENTENCE):
            word_train, word_valid = train[word], valid[word]
            for layer in range(n_layers):
                # create datasets / dataloaders
                layer_train = word_train.get_layer_dataset(layer) 
                layer_valid = word_valid.get_layer_dataset(layer)
                train_dataloader = DataLoader(layer_train, batch_size=batch_size, collate_fn=collate_fn_reps)
                valid_dataloader = DataLoader(layer_valid, batch_size=batch_size, collate_fn=collate_fn_reps)

                # initialize probe
                probe = Probe(model_config.hidden_size)
                
                # initialize path and gpu count
                path = Path(f'probes/{model_name}/run-{probe_run}/word-{word}/layer-{layer}')
                path.mkdir(exist_ok=True, parents=True)

                # train and save
                trainer = pl.Trainer(accelerator=pl_accelerator, devices=n_devices, max_epochs=epochs, default_root_dir=str(path))
                trainer.fit(probe, train_dataloader, valid_dataloader)
                torch.save(probe.state_dict(), path/'state_dict.pt')


def load_probe_params(model_name:str, probe_run:int, device:str='cpu'):
    """
    For a given model_name and probe run, loads the associated probes.
    """
    model_config = AutoConfig.from_pretrained(model_name)
    n_layers = model_config.num_hidden_layers + 1

    # the first model_config.hidden_size entries of the last dim are the weight
    # the last entry, at index model_config.hidden_size, is the bias
    probe_params = torch.zeros([WORDS_IN_SENTENCE, n_layers, model_config.hidden_size + 1])
    for word in range(WORDS_IN_SENTENCE):
        for layer in range(n_layers):
            path = Path(f'probes/{model_name}/run-{probe_run}/word-{word}/layer-{layer}/state_dict.pt')
            state_dict = torch.load(path)
            probe_params[word, layer, :model_config.hidden_size] = state_dict['probe.weight']
            probe_params[word, layer, model_config.hidden_size] = state_dict['probe.bias']
    return probe_params.to(device)


def probe_check(model_name:str, probe_runs:int):
    """
    Checks: for a given model_name and number of probe runs,
    are there at least that many probes trained?
    """
    model_config = AutoConfig.from_pretrained(model_name)
    n_layers = model_config.num_hidden_layers + 1
    for probe_run in range(probe_runs):
        for word in range(WORDS_IN_SENTENCE):
            for layer in range(n_layers):
                path = Path(f'probes/{model_name}/run-{probe_run}/word-{word}/layer-{layer}/state_dict.pt')
                if not path.exists():
                    return False
    return True