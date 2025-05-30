import json
import os
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser(description="Compute semantic similarity for translated texts in a folder.")
parser.add_argument('folder_path', type=str, help="Path to the folder containing the translated JSON files.")
args = parser.parse_args()

# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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

            # Compute semantic similarity
            similarity_scores = {
                'src': util.pytorch_cos_sim(model.encode(en_texts['src'], convert_to_tensor=True), model.encode(translated_texts['src'], convert_to_tensor=True)).item(),
                'rephrase': util.pytorch_cos_sim(model.encode(en_texts['rephrase'], convert_to_tensor=True), model.encode(translated_texts['rephrase'], convert_to_tensor=True)).item(),
                'alt': util.pytorch_cos_sim(model.encode(en_texts['alt'], convert_to_tensor=True), model.encode(translated_texts['alt'], convert_to_tensor=True)).item(),
                'loc': util.pytorch_cos_sim(model.encode(en_texts['loc'], convert_to_tensor=True), model.encode(translated_texts['loc'], convert_to_tensor=True)).item(),
                'loc_ans': util.pytorch_cos_sim(model.encode(en_texts['loc_ans'], convert_to_tensor=True), model.encode(translated_texts['loc_ans'], convert_to_tensor=True)).item(),
                'new_question': util.pytorch_cos_sim(model.encode(en_texts["portability"]['New Question'], convert_to_tensor=True), model.encode(translated_texts['new_question'], convert_to_tensor=True)).item(),
                'new_answer': util.pytorch_cos_sim(model.encode(en_texts["portability"]['New Answer'], convert_to_tensor=True), model.encode(translated_texts['new_answer'], convert_to_tensor=True)).item(),
            }

            avg_similarity_score = sum(similarity_scores.values()) / len(similarity_scores)

            results.append({
                'original_texts': item['original_texts'],
                'translated_texts': translated_texts,
                'en_texts': en_texts,
                'similarity_scores': similarity_scores,
                'avg_similarity_score': avg_similarity_score
            })

        # Save similarity scores to a new file
        output_filename = f'{os.path.splitext(filename)[0]}_similarity_scores.json'
        output_filepath = os.path.join(args.folder_path, output_filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # Calculate and save the average similarity score for the file
        file_avg_similarity_score = sum(result['avg_similarity_score'] for result in results) / len(results)
        all_results.append({
            'filename': filename,
            'file_avg_similarity_score': file_avg_similarity_score
        })
        print(f"Processed {filename}, File Average Similarity Score: {file_avg_similarity_score}")

# Save all files' average similarity scores to a summary file
with open(os.path.join(args.folder_path, 'all_files_avg_similarity_scores.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

for result in all_results:
    print(f"Filename: {result['filename']}, Average Similarity Score: {result['file_avg_similarity_score']}")
