from sentence_transformers import SentenceTransformer
import pickle
import json
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(train_path+'.json', 'rt') as f_corpus:
    corpus_lines = json.load(f_corpus)

with open(test_path, 'rt') as f_query:
    query_lines = json.load(f_query)

corpus_sentences = []
query_sentences = []
corpus_ids = []
query_ids = []

for line in corpus_lines:
    if 'en' in line:
        corpus_sentences.append(line['en']['src'] + ' ' + line['en']['alt'])
        corpus_ids.append(line['en']['case_id'])
    else:
        corpus_sentences.append(line['src'] + ' ' + line['alt'])
        corpus_ids.append(line['case_id'])
        
    

for idx, line in enumerate(query_lines):
    query_sentences.append(line['en']['src'] + ' ' + line['en']['alt'])
    if 'case_id' in line['en']:
        query_ids.append(line['en']['case_id'])
    else:
        query_ids.append(idx)

query_embeddings = model.encode(query_sentences, show_progress_bar=True)
corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True)

with open(train_path+'.pk', 'wb') as f_out:
    pickle.dump({'query_sentences': query_sentences, 'query_embeddings': query_embeddings, 'query_ids': query_ids,
                'corpus_sentences': corpus_sentences, 'corpus_embeddings': corpus_embeddings, 'corpus_ids': corpus_ids,}, 
                f_out, protocol=pickle.HIGHEST_PROTOCOL)
