#!/bin/bash

# Declare a string array with type
# 'bert-large-cased' 'roberta-large' 
declare -a StringArray=('roberta-base' 'distilbert-base-cased' 'distilroberta-base' 'bert-large-cased' 'roberta-large' )

# Read the array values with space
for val in "${StringArray[@]}"; do
 echo $val
 # python run_probing_interventions.py --model_name $val
 # python run_mdl_probing_sklearn.py --model_name $val
 # python run_vinfo.py --model_name $val
 python analysis_and_plots.py --model_name $val
done