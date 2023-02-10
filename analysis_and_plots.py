import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='roberta-large')
args = parser.parse_args()

FONTSIZE=15
WORDS = ['ART1', 'ADJ', 'SUBJ', 'ADV', '[MASK]', 'ART2', 'OBJ']

def make_heatmap(p_error: np.array, intervention:str):
    _, n_layers = p_error.shape
    labels = ['embs'] + list(range(n_layers-1))
    p_errors = {}
    for i, word in enumerate(WORDS):
        p_errors[word] = p_error[i]

    p_errors['labels'] = labels
    df = pd.DataFrame.from_dict(p_errors)
    df = df.set_index('labels')

    ax = sns.heatmap(df, vmin=0, vmax=1, cbar_kws={'label': 'p_error'})
    ax.invert_yaxis()

    title = f"p_error induced by {intervention} for {args.model_name}"
    savefile = Path(f"visualizations/{args.model_name}")
    savefile.mkdir(exist_ok=True, parents=True)
    plt.title(title)
    plt.xlabel('word')
    plt.ylabel('layer')
    plt.savefig(savefile/f'{intervention}.png')
    plt.close()

def make_mdl_graph(mdl: np.array, mdl_unmasked: np.array, legend_below:bool=True):
    plt.style.use('seaborn-darkgrid')
    plt.figure()

    _, n_layers = mdl.shape
    d = {word: np.minimum(mdl[i], 4000) for i, word in enumerate(WORDS)}
    d['VERB'] = np.minimum(mdl_unmasked[WORDS.index('[MASK]')], 4000)
    labels = ['embs'] + list(range(n_layers-1))
    d['layer'] = labels
    df = pd.DataFrame.from_dict(d)
    df = df.set_index('layer')

    ax = df.plot(kind='line', alpha=0.7)
    title = f"MDL by layer for {args.model_name}"
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel('Layer', fontsize=FONTSIZE)
    ax.set_ylabel('Codelength', fontsize=FONTSIZE)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(FONTSIZE)

    if legend_below:
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.17)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=4, fontsize=(FONTSIZE-2))
    else:
        ax.legend(loc='lower right', ncol=2, fontsize=FONTSIZE)

    
    savefile = Path(f"visualizations/{args.model_name}")
    savefile.mkdir(exist_ok=True, parents=True)
    fig = ax.get_figure()
    fig.savefig(savefile/'mdl.png')
    plt.close()

def make_vinfo_graph(accuracy:np.array, vinfo: np.array, accuracy_unmasked:np.array, vinfo_unmasked:np.array, legend_below:bool=True):
    plt.style.use('seaborn-darkgrid')
    plt.figure()

    _, n_layers = accuracy.shape
    accs = {word:accuracy[i] for i, word in enumerate(WORDS)}
    accs['VERB'] = accuracy_unmasked[WORDS.index('[MASK]')]
    labels = ['embs'] + list(range(n_layers-1))
    accs['layer'] = labels
    df = pd.DataFrame.from_dict(accs)
    df = df.set_index('layer')

    ax = df.plot(kind='line')
    title = f"Accuracy and V-Information by layer for {args.model_name}"
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel('Layer', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)

    vinfos = {word:vinfo[i] for i, word in enumerate(WORDS)}
    vinfos['VERB'] = vinfo_unmasked[WORDS.index('[MASK]')]
    df_v = pd.DataFrame.from_dict(vinfos)

    ax2 = ax.twinx()
    ax2.set_ylabel('V-Information', fontsize=FONTSIZE)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, column in enumerate(df_v.columns):
        ax2.plot(df_v[column], linestyle='--', label = '_nolegend_', color=color_cycle[i % len(color_cycle)])
    
    if legend_below:
        plt.subplots_adjust(bottom=0.25)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=4, fontsize=FONTSIZE)
    else:
        plt.subplots_adjust(bottom=0.13, right=0.87)
        ax.legend(loc='lower right', ncol=2, fontsize=FONTSIZE)

    for item in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels():
        item.set_fontsize(FONTSIZE)

    savefile = Path(f"visualizations/{args.model_name}")
    savefile.mkdir(exist_ok=True, parents=True)
    fig = ax.get_figure()
    fig.savefig(savefile/'v_info.png')
    plt.close()


# -------------------------- MAIN BODY STARTS HERE -------------------------- #
results_path = Path(f'results/{args.model_name}')

if (results_path/'interchange.pt').exists():
    # len(dataset), WORDS_TO_INTERVENE, n_layers, correctness (2)
    interchange = torch.load(results_path/'interchange.pt').numpy()
    disagreement = interchange[:, :, :, 1]
    mean_disagreement = disagreement.mean(0)
    std_disagreement = disagreement.std(0)
    make_heatmap(mean_disagreement, 'interchange')
else:
    print("Couldn't find interchange data")


if (results_path/'reflection.pt').exists():
    # probe_runs, len(dataset), WORDS_TO_INTERVENE, n_layers, correctness (2)
    reflection = torch.load(results_path/'reflection.pt').numpy()
    disagreement = reflection[:, :, :, :, 1]
    mean_disagreement = disagreement.mean(1)
    std_disagreement = disagreement.std(1)
    mean_disagreement_over_runs = mean_disagreement.mean(0)
    std_disagreement_over_runs = mean_disagreement.std(0)
    make_heatmap(mean_disagreement_over_runs, 'reflection')
else:
    print("Couldn't find reflection data")

if (results_path/'description_length.pt').exists():
    # probe_runs, WORDS_IN_SENTENCE, n_layers
    mdl = torch.load(results_path/'description_length.pt').numpy()
    mean_mdl = mdl.mean(0)
    std_mdl = mdl.std(0)
    mdl_unmasked = torch.load(results_path/'description_length_unmasked.pt').numpy()
    mean_mdl_unmasked = mdl_unmasked.mean(0)
    make_mdl_graph(mean_mdl, mean_mdl_unmasked)
else:
    print("Couldn't find MDL data.")

if (results_path/'accuracy.pt').exists() and (results_path/'v_info.pt').exists():
    # splits (3) x probe_runs x WORDS_IN_SENTENCE x n_layers
    accuracy = torch.load(results_path/'accuracy.pt').numpy()
    vinfo = torch.load(results_path/'v_info.pt').numpy()

    accuracy_unmasked = torch.load(results_path/'accuracy_unmasked.pt').numpy()
    vinfo_unmasked = torch.load(results_path/'v_info_unmasked.pt').numpy()

    train_accuracy, _, _ = accuracy 
    mean_train_accuracy = train_accuracy.mean(0)
    std_train_accuracy = train_accuracy.std(0)

    train_vinfo, _, _ = vinfo 
    mean_train_vinfo = train_vinfo.mean(0)
    std_train_vinfo = train_vinfo.std(0)

    make_vinfo_graph(mean_train_accuracy, mean_train_vinfo, accuracy_unmasked[0].mean(0), vinfo_unmasked[0].mean(0))
else:
    print("Couldn't find accuracy / V-info data")
