import json
import os
from tqdm import tqdm
import pandas as pd
from transformers import LlamaTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Coun', 'Wiki', 'zsRE'],
                       help='Dataset type to filter (Coun/Wiki/zsRE)')
    return parser.parse_args()

def parse_filename(filename):
    # Split filename to get components
    parts = filename.replace('.json', '').split('_')
    
    # Extract method (FT/ROME/LoRA/KN)
    method = parts[0]
    
    # Extract dataset type (Coun/Wiki/zsRE)
    dataset = parts[1]
    
    # Extract language code
    lang = parts[2]
    
    # Extract result number
    result_num = parts[-1]
    
    return method, dataset, lang, result_num

def filter_files(files, target_dataset):
    filtered_files = []
    for f in files:
        if not f.endswith('.json'):
            continue
            
        method, dataset, lang, result_num = parse_filename(f)
        
        # Only include files from specified dataset ending with _0.json
        if dataset == target_dataset and result_num == '0':
            filtered_files.append(f)
            
    return filtered_files

def organize_results(results_dict):
    # Get unique methods from the results
    methods = sorted(set(k[1] for k in results_dict.keys()))
    metrics = ['Reliability', 'Generality', 'Locality', 'Portability']
    
    # Initialize DataFrame
    index = pd.MultiIndex.from_product([methods, metrics], 
                                     names=['Method', 'Metric'])
    columns = []
    
    # Sort results by language and metric type (F1/EM)
    for lang in sorted(set(k[0] for k in results_dict.keys())):
        columns.extend([f'{lang}_F1', f'{lang}_EM'])
        
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame
    for (lang, method, metric) in results_dict:
        f1_score, em_score = results_dict[(lang, method, metric)].split('/')
        df.loc[(method, metric), f'{lang}_F1'] = float(f1_score)
        df.loc[(method, metric), f'{lang}_EM'] = float(em_score)
        
    return df

def obtain_f1_and_em(a, b):
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em

def my_avg(a):
    return round(sum(a) * 100 / float(len(a)), 2)

def calculate_metrics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化度量存储
    reliability_f1, reliability_em = [], []
    generalization_f1, generalization_em = [], []
    locality_f1, locality_em = [], []
    portability_f1, portability_em = [], []

    print(f"Processing {len(data)} items from {file_path}...")  # 打印数据量

    for item in tqdm(data, desc="Processing data", unit="item"):  
        # Reliability
        f1, em = obtain_f1_and_em(item["post"]["reliability"]["ans"], item["post"]["reliability"]["target"])
        reliability_f1.append(f1)
        reliability_em.append(em)

        # Generalization
        f1, em = obtain_f1_and_em(item["post"]["generalization"]["ans"], item["post"]["generalization"]["target"])
        generalization_f1.append(f1)
        generalization_em.append(em)

        # Locality
        f1, em = obtain_f1_and_em(item["post"]["locality"]["neighborhood_acc"]["ans"],
                                  item["pre"]["locality"]["neighborhood_acc"]["ans"])
        locality_f1.append(f1)
        locality_em.append(em)

        # Portability
        f1, em = obtain_f1_and_em(item["post"]["portability"]["one_hop_acc"]["ans"],
                                  item["post"]["portability"]["one_hop_acc"]["target"])
        portability_f1.append(f1)
        portability_em.append(em)

    print("Processing complete!")  

    results = {
        "reliability": f"{my_avg(reliability_f1)}/{my_avg(reliability_em)}",
        "generalization": f"{my_avg(generalization_f1)}/{my_avg(generalization_em)}",
        "locality": f"{my_avg(locality_f1)}/{my_avg(locality_em)}",
        "portability": f"{my_avg(portability_f1)}/{my_avg(portability_em)}"
    }

    print("Processing avg complete!")  
    return results

def main():
    args = parse_args()
    path = "./output/"
    files = os.listdir(path)
    
    # Filter files based on dataset type and ending with _0
    filtered_files = filter_files(files, args.dataset)
    
    if not filtered_files:
        print(f"No files found for dataset {args.dataset}")
        return
        
    results_dict = {}
    for f in filtered_files:
        method, dataset, lang, _ = parse_filename(f)
        file_path = os.path.join(path, f)
        metrics = calculate_metrics(file_path)
        
        # Store results with language and metric type
        for metric_name, score in metrics.items():
            metric_map = {
                'reliability': 'Reliability',
                'generalization': 'Generality',
                'locality': 'Locality',
                'portability': 'Portability'
            }
            results_dict[(lang, method, metric_map[metric_name])] = score
    
    # Organize results into DataFrame
    df = organize_results(results_dict)
    
    # Save results
    output_file = f'./csv-results/{args.dataset.lower()}_analysis_results.csv'
    df.to_csv(output_file)
    print(f"Results saved to {output_file}")
    
    # Display results
    print("\nResults Summary:")
    print(df)

if __name__ == "__main__":
    main()