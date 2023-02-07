# probed-information
This repository contains all of the code needed to replicate the 2023 EACL paper "The Functional Relevance of Probed Information: A Case Study".

To do so, simply run the three files corresponding to the three kinds of plot in the paper:
- `run_probing_interventions.py`
- `run_mdl_probing.py`
- `run_vinfo.py`

You will also need to specify the model that you wish to run this on, using the `--model_name` flag. Supported models are `{'bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'distilbert-base-cased', 'distilroberta-base'}`.

The packages needed to install this package can be installed together as a conda environement, using the `environment.yml` file.

Feel free to email me (m.w.hanna@uva.nl) with any questions!
