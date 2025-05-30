import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import random
from tqdm import tqdm
import os
import csv
import random


TYPE2DEMOS = {
    "rel_gen": [1,2,9,10,17,18,25,26],
    "loc": [5,6,13,14,21,22,29,30],
    "port": [7,8,15,16,23,24,31,32] 
}


def obtain_f1_and_em(a, b):
    global tokenizer
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
    if len(a) == 0:
        return 0
    else:
        return round(sum(a) * 100 / float(len(a)), 2)


def icl_lm_eval_f1em(model, tokenizer, icl_examples, target, x):   
    device = torch.device('cuda')
    
    inputs = tokenizer(''.join(icl_examples) + x, return_tensors='pt').to(device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=15, 
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
    content = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    textual_ans = content.split('"')[0]
    return textual_ans

def icl_lm_eval_ppls(model, tokenizer, icl_examples, targets, x):
    device = torch.device('cuda')
    ppls = [] 
    for target in targets:
        tgt_len = len(tokenizer.encode(target, add_special_tokens=False))
        encodings = tokenizer(''.join(icl_examples) + f'{x}{target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
    return ppls

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument("--lang1", type=str, default="")
    parser.add_argument("--lang2", type=str, default="")
    parser.add_argument("--manualdata", type=str, default="")
    parser.add_argument("--testdata", type=str, default="")
    parser.add_argument("--lcount", type=int, default=3000)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B", 
                        choices=["meta-llama/Llama-3.2-3B", "meta-llama/Meta-Llama-3.1-8B", "bigscience/bloomz-7b1", "mistralai/Ministral-8B-Instruct-2410", "Qwen/Qwen2.5-7B-Instruct"])
    parser.add_argument("--api_key_path", type=str, default="api-keys/hf_key.txt")
    parser.add_argument('--pbase', action='store_true', help='Prompt baseline')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--demo_consistent', action="store_true", help="same ICL demo metric types")
    args = parser.parse_args()
    return args

def construct_icl_examples(args): 
    icl_examples = []
    with open(f'./data/manual_prompts/{args.manualdata}.json', 'r') as fIn: # mcounterfact_multi   zsre_multi   wfd_multi
        lines = json.load(fIn)
        if args.one_shot:
            line = lines[random.randint(0,7)]
            lang1 = line['new_fact'] if args.lang1 == 'en' else args.lang1
            return [f"{lang1}\n{line[args.lang2]}\n\n"]
        
        if args.demo_consistent:
            icl_examples_rel_gen, icl_examples_loc, icl_examples_port = [], [], []
            for metric, idxs in TYPE2DEMOS.items():
                for idx in idxs:
                    line = lines[idx-1]
                    lang1 = line['new_fact'] if args.lang1 == 'en' else args.lang1
                    demo = f"{lang1}\n{line[args.lang2]}\n\n"
                    if metric == "rel_gen":
                        icl_examples_rel_gen.append(demo)
                    elif metric == "loc":
                        icl_examples_loc.append(demo)
                    elif metric == "port":
                        icl_examples_port.append(demo)
                    else:
                        print(f"The type \"{metric}\" not in manual prompts.")
            
            return icl_examples_rel_gen, icl_examples_loc, icl_examples_port
                    
        
        for line in lines[:8]:
            lang1 = line['new_fact'] if args.lang1 == 'en' else args.lang1
            icl_examples.append(f"{lang1}\n{line[args.lang2]}\n\n")
    icl_examples.reverse()
    return icl_examples

def wrap_f1em_list(listf1, listem, ans, target):
    single_f1, single_em = obtain_f1_and_em(ans, target)
    listf1.append(single_f1)
    listem.append(single_em)

def wrap_ppls_count(edit_ppls, total_cnt, success_cnt, magnitude ):
    edit_final_probs = [1 / edit_ppls[0], 1 / edit_ppls[1]]
    total_cnt += 1
    if edit_final_probs[0] > edit_final_probs[1]:
        success_cnt += 1
    magnitude += edit_final_probs[0] - edit_final_probs[1]
    return total_cnt, success_cnt, magnitude

if __name__ == '__main__':
    args = parse_args()
    device = args.device
    model_name = args.model_name

    with open(args.api_key_path, "r") as f_oa:
        hf_token = f_oa.readline().strip()
        
    random.seed(args.seed)
    
    if args.pbase:
        args.num_icl = 0
    elif args.one_shot:
        args.num_icl = 1
    elif args.demo_consistent:
        args.num_icl = "8a"
    else:
        args.num_icl = 8
                
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    lines = []
    with open(f'./data/{args.testdata}{args.lang2}.json', 'r') as f:
        lines = json.load(f)
        # lines = json.load(f)[:10] # Test
        
    icl_examples, icl_examples_rel_gen, icl_examples_port, icl_examples_loc = [], [], [], []
    success_cnt = 0
    para_success_cnt = 0
    magnitude = .0
    para_magnitude = .0
    orig_magnitude = .0
    total_cnt = 0
    para_total_cnt = 0
    orig_success_cnt = 0
    orig_total_cnt = 0
    reliablilty_f1_list = []
    reliablilty_em_list = []
    generalization_f1_list = []
    generalization_em_list = []

    locality_f1_list = []
    locality_em_list = []
    specificity_f1_list = []
    specificity_em_list = []

    portablility_f1_list = []
    portablility_em_list = []

    # Print Switch
    f1r,f1g,f1l,f1p,pplr,pplg,ppll = False,False,False,False,False,False,False

    results = []
    
    example_idx = 0
    if not args.pbase:
        icl_examples = construct_icl_examples(args)
        icl_examples_rel_gen, icl_examples_loc, icl_examples_port = icl_examples, icl_examples, icl_examples
    if args.demo_consistent:
        icl_examples_rel_gen, icl_examples_loc, icl_examples_port = construct_icl_examples(args)
    for i, line in enumerate(tqdm(lines[:args.lcount], total=len(lines[:args.lcount]), desc="Processing lines")):
        if args.lang2 not in line:
            continue
        case_id = line[args.lang1]['case_id']
        subject = line[args.lang1]['subject']
        prompts_truth = line[args.lang1]['src']
        prompts_test = line[args.lang2]['src']

        target_truth = line[args.lang1]['alt']
        target_test = line[args.lang2]['alt']

        rephrase_prompt = line[args.lang2]['rephrase']
        locality_prompt = line[args.lang2]['loc']
        locality_an = line[args.lang2]['loc_ans']
        portability_prompt = line[args.lang2]['port']
        portability_an = line[args.lang2]['port_ans']

        
        # reliablilty (f1em)
        input_text = f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{prompts_test}" Answer: "'
        ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_rel_gen, target_test, input_text)
        results.append([case_id, "reliability", input_text, ans, target_test])
        wrap_f1em_list(reliablilty_f1_list, reliablilty_em_list, ans, target_test)

        # generalization (f1em)
        input_text = f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{rephrase_prompt}" Answer: "'
        ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_rel_gen, target_test, input_text)
        results.append([case_id, "generality", input_text, ans, target_test])
        wrap_f1em_list(generalization_f1_list, generalization_em_list, ans, target_test)

        # locality (f1em)
        input_text = f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{locality_prompt}" Answer: "'
        ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_loc, locality_an, input_text)
        results.append([case_id, "locality", input_text, ans, locality_an])
        wrap_f1em_list(locality_f1_list, locality_em_list, ans, locality_an)

        # portablility (f1em)
        input_text = f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{portability_prompt}" Answer: "'
        ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_port, portability_an, input_text)
        results.append([case_id, "portability", input_text, ans, portability_an])
        wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)
        
        if "zsRE" in args.testdata:
            f1r,f1g,f1l,f1p = True,True,True,True
            
        elif  "Counter" in args.testdata or "WikiFact" in args.testdata:
            # reliablilty (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples_rel_gen, [target_test, locality_an], f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{prompts_test}" Answer: "')
            orig_total_cnt, orig_success_cnt, orig_magnitude = wrap_ppls_count(edit_ppls, orig_total_cnt, orig_success_cnt, orig_magnitude)

            # generalization (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples_rel_gen, [target_test, locality_an], f'New Fact: "{prompts_truth} {target_truth}"\nQuestion: "{rephrase_prompt}" Answer: "')
            para_total_cnt, para_success_cnt, para_magnitude = wrap_ppls_count(edit_ppls, para_total_cnt, para_success_cnt, para_magnitude)


            # print(f"success_cnt: {success_cnt}, total_cnt {total_cnt}")

            f1r,f1g,f1l,f1p,pplr,pplg = True,True,True,True,True,True
            
        else:
            print("unvalid test data")

        example_idx += 1
        # print(example_idx)

    # print the result
    print("F1EM score")
    if f1r: print("reliablilty_f1: %f   reliablilty_em: %f" % (my_avg(reliablilty_f1_list), my_avg(reliablilty_em_list)))
    if f1g: print("generalization_f1: %f    generalization_em: %f" % (my_avg(generalization_f1_list), my_avg(generalization_em_list)))
    if f1l: print("locality_f1: %f  locality_em: %f" % (my_avg(locality_f1_list), my_avg(locality_em_list)))
    if f1p: print("portablility_f1: %f  portablility_em: %f" % (my_avg(portablility_f1_list), my_avg(portablility_em_list)))

    print("PPLS score")
    if pplr: print("reliablilty_ppls: %f, magnitude: %f" % (orig_success_cnt/orig_total_cnt*100, orig_magnitude/orig_total_cnt*100))
    if ppll: print("locality_ppls: %f, magnitude: %f" % (success_cnt/total_cnt*100, magnitude/total_cnt*100))
    if pplg: print("generalization_ppls: %f, magnitude: %f" % (para_success_cnt/para_total_cnt*100, para_magnitude/para_total_cnt*100))


    # write to the result file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name = f'output/{args.model_name.split("/")[-1]}/icl{args.num_icl}_{args.testdata}_{args.lang1}{args.lang2}.txt'
    output_file_path = os.path.join(root_dir, output_file_name)
    output_folder = os.path.dirname(output_file_path)
    os.makedirs(output_folder, exist_ok=True)
    
    result_file_name = f'result/{args.model_name.split("/")[-1]}/icl{args.num_icl}_{args.testdata}_{args.lang1}{args.lang2}.csv'
    result_file_path = os.path.join(root_dir, result_file_name)
    result_folder = os.path.dirname(result_file_path)
    os.makedirs(result_folder, exist_ok=True)
    
    with open(result_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "metric", "input", "prediction", "true"])
        writer.writerows(results)

    with open(output_file_path, 'w+') as f:
        f.write("F1EM score\n")
        if f1r:
            f.write(f"reliability_f1: {my_avg(reliablilty_f1_list):.6f}   reliability_em: {my_avg(reliablilty_em_list):.6f}\n")
        if f1g:
            f.write(f"generalization_f1: {my_avg(generalization_f1_list):.6f}   generalization_em: {my_avg(generalization_em_list):.6f}\n")
        if f1l:
            f.write(f"locality_f1: {my_avg(locality_f1_list):.6f}   locality_em: {my_avg(locality_em_list):.6f}\n")
        if f1p:
            f.write(f"portability_f1: {my_avg(portablility_f1_list):.6f}   portability_em: {my_avg(portablility_em_list):.6f}\n")
        
        f.write("\nPPLS score\n")
        if pplr and orig_total_cnt != 0:
            f.write(f"reliability_ppls: {orig_success_cnt/orig_total_cnt*100:.6f}, magnitude: {orig_magnitude/orig_total_cnt*100:.6f}\n")
        if ppll and total_cnt != 0:
            f.write(f"locality_ppls: {success_cnt/total_cnt*100:.6f}, magnitude: {magnitude/total_cnt*100:.6f}\n")
        if pplg and para_total_cnt != 0:
            f.write(f"generalization_ppls: {para_success_cnt/para_total_cnt*100:.6f}, magnitude: {para_magnitude/para_total_cnt*100:.6f}\n")