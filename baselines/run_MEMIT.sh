#!/bin/bash

LANGS=("vi" ) #"es"  "ru" "zh" "es"
DATAS=("zsRE/zsre_test_" "CounterFact/counterfact_test_" "WikiFactDiff/wfd_test_")
CUDA=0

for DATA in "${DATAS[@]}";do
    for LANG in "${LANGS[@]}";do
        echo "currently processing language: $LANG with data: $DATA"
        python run_zsre_llama2.py --lang1 en --lang2 $LANG --editing_method MEMIT --hparams_dir ./hparams/MEMIT/llama3.2-3b.yaml  --data_dir $DATA
    done
done
# CUDA_VISIBLE_DEVICES=$CUDA 