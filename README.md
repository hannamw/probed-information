# probed-information
This repository contains all of the code needed to replicate the 2023 EACL paper "The Functional Relevance of Probed Information: A Case Study".

To do so, simply run the following commands, replacing <model_name> with a model in `{'bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'distilbert-base-cased', 'distilroberta-base'}`.
```
python run_probing_interventions.py --model_name bert-base-cased --batch_size 16
python run_mdl_probing.py --model_name bert-base-cased --batch_size 16
python run_mdl_probing.py --model_name bert-base-cased --batch_size 16 --unmasked
python run_vinfo.py --model_name bert-base-cased --batch_size 16
python run_vinfo.py --model_name bert-base-cased --batch_size 16 --unmasked
python analysis_and_plots.py --model_name bert-base-cased
```
The packages needed to install this package can be installed together as a conda environement, using the `environment.yml` file.

Feel free to email me (m.w.hanna@uva.nl) with any questions!
