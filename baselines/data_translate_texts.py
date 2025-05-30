import json
import os
import time
import deepl
import argparse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "transpdf-6520edf19d33.json"

# DeepL key
auth_key = "4d7cbf44-8015-4de4-868d-0f9b4668f17c:fx" 
deepl_translator = deepl.Translator(auth_key)
# Function to translate text
def deepl_translate_text(text, dest_lang='EN-US'):
    try:
        time.sleep(0.1)
        translation = deepl_translator.translate_text(text, target_lang=dest_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return ""


def google_translate_text(text, dest_lang='EN-US') -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=dest_lang)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result["translatedText"]


# Argument parser setup
parser = argparse.ArgumentParser(description="Translate texts from files in a folder.")
parser.add_argument('folder_path', type=str, help="Path to the folder containing the JSON files.")
parser.add_argument('output_folder_path', type=str, help="Path to the folder where the translated files will be saved.")
args = parser.parse_args()


# List of target languages
lang_list = ["es", "vi", "ru", "zh-cn", "de"]

# Process files in the folder
for filename in os.listdir(args.folder_path):
    if 'test' in filename and not 'enen' in filename:
        file_path = os.path.join(args.folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        lang2_code = list(data[0].keys())[1]
        if lang2_code not in lang_list: 
            print(f"Skipping file with unsupported language code: {lang2_code}")
            continue
        
        for item in data[:50]:
            lang2_texts = item[lang2_code]
            translated_texts = {
                'src': google_translate_text(lang2_texts['src']),
                'rephrase': google_translate_text(lang2_texts['rephrase']),
                'alt': google_translate_text(lang2_texts['alt']),
                'loc': google_translate_text(lang2_texts['loc']),
                'loc_ans': google_translate_text(lang2_texts['loc_ans']),
                'new_question': google_translate_text(lang2_texts['portability']['New Question']),
                'new_answer': google_translate_text(lang2_texts['portability']['New Answer']),
            }
            results.append({
                'original_texts': lang2_texts,
                'translated_texts': translated_texts,
                'en_texts': item['en']
            })

        # Save translated results to a new file
        output_filename = f'{os.path.splitext(filename)[0]}_translated.json'
        output_filepath = os.path.join(args.output_folder_path, output_filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Translated texts saved to {output_filename}")
