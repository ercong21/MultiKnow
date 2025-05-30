#!/bin/bash

LANGS=("ar" "he" "fa" "de" "fr" "zh" "ja" "hu" "ru" "tr") #"vi" "es"  "ru" "zh"  "es" # "ar" "he" "fa" "de" "fr" "zh" "ja" "hu" "ru" "tr"
DATAS=("zsRE/zsre_test_") # "zsRE/zsre_test_" "CounterFact/counterfact_test_"  "WikiFactDiff/wfd_test_"
CUDA=1

for DATA in "${DATAS[@]}";do
    for LANG in "${LANGS[@]}";do
        echo "currently processing language: $LANG with data: $DATA"
        CUDA_VISIBLE_DEVICES=$CUDA python run_zsre_llama2.py --lang1 en --lang2 $LANG --editing_method KN --hparams_dir ./hparams/KN/llama3.2-3b.yaml --data_dir $DATA 
    done
done

# --ds_size 200 