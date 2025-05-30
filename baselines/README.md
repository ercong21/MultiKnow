# Multilingual Knowledge Editing (Baseline)

This directory includes implementations of baseline methods used for evaluating and comparing editing performance.

---

## 📦 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```


## 🧪 Run a Single Experiment
This script applies a chosen editing method (e.g., ROME, LoRA, FT) to our multilingual knowledge editing dataset. 

Example: Run ROME on zsRE dataset (from English to Vietnamese):

```bash
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python run_zsre_llama2.py \
  --editing_method ROME \
  --hparams_dir ./hparams/ROME/llama3.2-3b \
  --data_dir zsRE/zsre_test_ \
  --lang1 en \
  --lang2 vi" > ./logs/output_ROME.log 2>&1 &
```

## 🚀 Batch Execution of Each Editing Method Across Languages
These scripts run knowledge editing across multiple target languages and different subsets of datasets using various methods: FT, KN, ROME, MEND, and LoRA.

Make scripts executable:
 
```bash
chmod +x run_FT.sh
chmod +x run_KN.sh
chmod +x run_ROME.sh
chmod +x run_MEND.sh
chmod +x run_LoRA.sh
```
Run each method:
```bash
./run_FT.sh
./run_KN.sh
./run_ROME.sh
./run_MEND.sh
./run_LoRA.sh
```


## 📝 Run Scripts with Logging

Ensure CUDA_VISIBLE_DEVICES is set within each script before running via nohup.
 
```bash
nohup ./run_ROME.sh > ./logs/output_ROME.log 2>&1 &
nohup ./run_KN.sh > ./logs/output_KN.log 2>&1 &
nohup ./run_LoRA.sh > ./logs/output_LoRA.log 2>&1 &
nohup ./run_FT.sh > ./logs/output_FT.log 2>&1 &
```

---

## ✅ Evaluation

Run evaluation for a specific dataset (e.g., Coun):
 
```bash
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python evaluate_test.py --dataset Coun" > ./logs/output_evaluate.log 2>&1 &
```

---

## ✅ Dataset Quality Evaluation
Translate texts:

```bash
python data_translate_texts.py ./data/MzsRE ./data/MzsRE/result
```
Compute BLEU scores:

```bash
python data_compute_bleu_scores.py ./data/MzsRE/result
```
Compute semantic similarity:

```bash
python data_compute_semantic_similarity.py ./data/MzsRE/result
```



## 📊 Monitoring Commands

```bash
nvidia-smi                  # Monitor GPU usage
ps aux | grep python        # Check running Python processes
```

---

## 🙏 Acknowledgement
We gratefully acknowledge the authors of [EasyEdit](https://github.com/zjunlp/EasyEdit), upon which our experiments are built. Their open-source framework has provided a strong foundation for this work.

---

## License
This project is licensed under the [MIT License](LICENSE).