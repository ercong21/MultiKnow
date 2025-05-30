import json
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import LlamaTokenizer, AutoTokenizer
import numpy as np

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def obtain_f1_and_em(a, b):
    global tokenizer

    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    # if len(a_words) == 0 and len(b_words) == 0:
    #     return 1.0, 1
    # if len(a_words) == 0 or len(b_words) == 0:
    #     return 0.0, 0

    # em = 1 if a == b else 0
    # k = len(a_words) * len(b_words)

    # intersecting_words = []
    # for word in a_words.copy():
    #     if word in b_words:
    #         a_words.remove(word)
    #         b_words.remove(word)
    #         intersecting_words.append(word)
    # Pad the shorter array with a special value (-1)
    max_len = max(len(a_words), len(b_words))
    a_words = np.pad(a_words, (0, max_len - len(a_words)), constant_values=-1)
    b_words = np.pad(b_words, (0, max_len - len(b_words)), constant_values=-1)

    print(a_words)
    print(b_words)
    f1 = f1_score(a_words, b_words, average='macro', zero_division=1)
    em = np.mean(np.equal(a_words, b_words))
    return f1, em


def my_avg(a):
    return round(sum(a) * 100 / float(len(a)), 2)


def calculate_metrics(file_root):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    for item in tqdm(data):
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(item["post"]["reliability"]["ans"],
                                                          item["post"]["reliability"]["target"])
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        generalization_f1, generalization_em = obtain_f1_and_em(item["post"]["generalization"]["ans"],
                                                                item["post"]["generalization"][
                                                                    "target"])
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)

        locality_f1, locality_em = obtain_f1_and_em(item["post"]["locality"]["neighborhood_acc"]["ans"],
                                                          item["pre"]["locality"]["neighborhood_acc"]["ans"])
        locality_f1_list.append(locality_f1)
        locality_em_list.append(locality_em)


        portablility_f1, portablility_em = obtain_f1_and_em(item["post"]["portability"]["one_hop_acc"]["ans"],
                                                            item["post"]["portability"]["one_hop_acc"]["target"])
        portablility_f1_list.append(portablility_f1)
        portablility_em_list.append(portablility_em)



    print("=" * 20 + file_root + "=" * 20)
    print("F1 score")
    print("reliablilty_f1: %f" % (my_avg(reliablilty_f1_list)))
    print("generalization_f1: %f" % my_avg(generalization_f1_list))
    print("locality_f1: %f"%my_avg(locality_f1_list))
    print("portablility_f1: %f" % my_avg(portablility_f1_list))

    print("EM score")
    print("reliablilty_em: %f" % (my_avg(reliablilty_em_list)))
    print("generalization_em: %f" % my_avg(generalization_em_list))
    print("locality_em: %f"%my_avg(locality_em_list))
    print("portablility_em: %f" % my_avg(portablility_em_list))

    reli, gene, loca, port = str(my_avg(reliablilty_f1_list)) + '/' + str(my_avg(reliablilty_em_list)),str(my_avg(generalization_f1_list)) + '/' + str(my_avg(generalization_em_list)),str(my_avg(locality_f1_list)) + '/' + str(my_avg(locality_em_list)),str(my_avg(portablility_f1_list)) + '/' + str(my_avg(portablility_em_list))


    return reli, gene, loca, port


if __name__ == "__main__":

    path = "./output/"
    out_f = open(
        "./csv-results/llama7-easyedit-16.csv",
        "w", encoding="utf-8")

    files = os.listdir(path)
    out_f.write(
        'lang' + '\t' + 'reliability' + '\t' + 'generalization' + '\t' + 'locality' + '\t' + 'portability' + '\n')
    for f in files:
        if f.endswith('json'):
            file = path + "/" + f
            reli, gene, loca, port = calculate_metrics(file)
            out_f.write(f + '\t' + reli + '\t' + gene + '\t' + loca + '\t' + port + '\n')
    out_f.close()
