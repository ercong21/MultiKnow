import json
import os
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download punkt for tokenization
nltk.download('punkt')

# Function to compute BLEU score
def compute_bleu_score(reference, hypothesis, n_gram=1):
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)

    weights = [1.0 / n_gram] * n_gram
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], hypothesis_tokens, weights=weights, smoothing_function=smoothing_function)

# Argument parser setup
parser = argparse.ArgumentParser(description="Compute BLEU scores for translated texts in a folder.")
parser.add_argument('folder_path', type=str, help="Path to the folder containing the translated JSON files.")
args = parser.parse_args()

# Process translated files in the folder
all_results = []
for filename in os.listdir(args.folder_path):
    if filename.endswith('_translated.json'):
        file_path = os.path.join(args.folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data:
            en_texts = item['en_texts']
            translated_texts = item['translated_texts']

            # Compute BLEU scores
            bleu_scores = {
                'src': compute_bleu_score(en_texts['src'], translated_texts['src']),
                'rephrase': compute_bleu_score(en_texts['rephrase'], translated_texts['rephrase']),
                'alt': compute_bleu_score(en_texts['alt'], translated_texts['alt']),
                'loc': compute_bleu_score(en_texts['loc'], translated_texts['loc']),
                'loc_ans': compute_bleu_score(en_texts['loc_ans'], translated_texts['loc_ans']),
                'new_question': compute_bleu_score(en_texts["portability"]['New Question'], translated_texts['new_question']),
                'new_answer': compute_bleu_score(en_texts["portability"]['New Answer'], translated_texts['new_answer']),
            }

            avg_bleu_score = sum(bleu_scores.values()) / len(bleu_scores)

            results.append({
                'original_texts': item['original_texts'],
                'translated_texts': translated_texts,
                'en_texts': en_texts,
                'bleu_scores': bleu_scores,
                'avg_bleu_score': avg_bleu_score
            })

        # Save BLEU scores to a new file
        output_filename = f'{os.path.splitext(filename)[0]}_bleu_scores.json'
        output_filepath = os.path.join(args.folder_path, output_filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # Calculate and save the average BLEU score for the file
        file_avg_bleu_score = sum(result['avg_bleu_score'] for result in results) / len(results)
        all_results.append({
            'filename': filename,
            'file_avg_bleu_score': file_avg_bleu_score
        })
        print(f"Processed {filename}, File Average BLEU Score: {file_avg_bleu_score}")

# Save all files' average BLEU scores to a summary file
with open(os.path.join(args.folder_path, 'all_files_avg_bleu_scores.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

for result in all_results:
    print(f"Filename: {result['filename']}, Average BLEU Score: {result['file_avg_bleu_score']}")
