# BMIKE-53: Cross-Lingual In-Context Knowledge Editing Benchmark


[![Paper (ACL 2025)](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://arxiv.org/pdf/2406.17764)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the code and data for the ACL 2025 paper: "[*BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning*](https://arxiv.org/pdf/2406.17764)"

## Overview
**BMIKE-53** is the first comprehensive multilingual benchmark for **Cross-Lingual In-Context Knowledge Editing (IKE)**. It covers 53 languages and unifies three widely used knowledge editing datasets (zsRE, CounterFact, WikiFactDiff) into a consistent, multilingual format. This resource enables systematic evaluation and analysis of knowledge editing capabilities in large language models (LLMs) across a broad spectrum of languages and knowledge scenarios.

## Table of Contents

- [Datasets](#datasets)
- [Supported Languages](#supported-languages)
- [Benchmark Structure](#benchmark-structure)
- [How to Use](#how-to-use)
- [Citation](#citation)
- [License](#license)

---

## Datasets

- **zsRE:** Zero-shot relation extraction, regular fact modifications.
- **CounterFact:** Counterfactual (fabricated) fact edits for testing knowledge locality.
- **WikiFactDiff:** Real-world, temporally dynamic factual updates derived from WikiData.

---

## Supported Languages

BMIKE-53 covers 53 languages, including:

`af`, `ar`, `az`, `be`, `bg`, `bn`, `ca`, `ceb`, `cs`, `cy`, `da`, `de`, `el`, `es`, `et`, `eu`, `fa`, `fi`, `fr`, `ga`, `gl`, `he`, `hi`, `hr`, `hu`, `hy`, `id`, `it`, `ja`, `ka`, `ko`, `la`, `lt`, `lv`, `ms`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sq`, `sr`, `sv`, `ta`, `th`, `tr`, `uk`, `ur`, `vi`, `zh`, `en`.

See [data/lang.json](data/lang.json) for details.

---

## Benchmark Structure

### Data Format

Each BMIKE-53 data sample is structured as a dictionary with entries for each language (e.g., `"en"` for English, `"de"` for German, etc.). Each language-specific entry contains:

- `case_id`: Unique identifier for the knowledge edit case.
- `subject`: The subject entity or person the knowledge edit is about.
- `src`: The original query (reliability type), directly matching the edited knowledge.
- `rephrase`: A paraphrased version of the original query, testing generalization.
- `old`: The original (pre-edit) answer to the query.
- `alt`: The updated (post-edit) answer reflecting the new knowledge.
- `loc`: A locality control query about an unrelated subject, testing if unrelated knowledge is preserved.
- `loc_ans`: The correct answer for the locality query.
- `port`: A portability query about a related (one-hop) aspect, testing the transfer of edited knowledge.
- `port_ans`: The correct answer for the portability query.

All values are language-specific translations, and the same structure is repeated for each supported language.

**Example:**
```json
{
  "en": {
    "case_id": 5,
    "subject": "Bhagwant Mann",
    "src": "What position did Bhagwant Mann hold?",
    "rephrase": "What role did Bhagwant Mann serve in?",
    "old": "Member of the 17th Lok Sabha",
    "alt": "Chief Minister of Punjab",
    "loc": "Who was Dipsinh Shankarsinh Rathod?",
    "loc_ans": "Member of the 17th Lok Sabha",
    "port": "Who is the current officeholder of the position previously held by Bhagwant Mann?",
    "port_ans": "Amarinder Singh"
  },
  "de": {
    "case_id": 5,
    "subject": "Bhagwant Mann",
    "src": "Welche Position hatte Bhagwant Mann inne?",
    "rephrase": "Welche Rolle hat Bhagwant Mann ausgeübt?",
    "old": "Mitglied des 17. Lok Sabha",
    "alt": "Chief Minister von Punjab",
    "loc": "Wer war Dipsinh Shankarsinh Rathod?",
    "loc_ans": "Mitglied des 17. Lok Sabha",
    "port": "Wer ist der aktuelle Amtsinhaber der Position, die zuvor von Bhagwant Mann gehalten wurde?",
    "port_ans": "Amarinder Singh"
  }
}
```

### Files

- Benchmark data is organized under the `/data/BMIKE53/` directory, with three subfolders corresponding to each dataset:  
  - `/CounterFact/`
  - `/WikiFactDiff/`
  - `/zsRE/`
- Each dataset folder contains test files for all 53 languages, named using the pattern:  
  - `zsre_test_{lang}.json` (e.g., `zsre_test_en.json`, `zsre_test_de.json`, etc.)
  - `counterfact_test_{lang}.json`
  - `wikifactdiff_test_{lang}.json`
- Each file contains the test set for one language in JSON format, following the unified BMIKE-53 data structure.
- Example:
  ```
  data/BMIKE53/zsRE/zsre_test_en.json
  data/BMIKE53/zsRE/zsre_test_de.json
  data/BMIKE53/CounterFact/counterfact_test_en.json
  data/BMIKE53/WikiFactDiff/wikifactdiff_test_en.json
  ```

---

## How to Use

### Download

Clone the repository and download the data:

```bash
git clone https://github.com/your-org/BMIKE-53.git
cd BMIKE-53
# Download data files, if not included in repo
```

Certainly! Here’s an updated **Benchmarking** section for your README, including clear command-line code examples for running BMIKE-53 experiments with different in-context learning setups:

---

### Benchmarking

1. **Select Task and Language(s)**
   - Choose the dataset (e.g., zsRE, CounterFact, WikiFactDiff) and specify the source (`--lang1`) and target (`--lang2`) languages.
2. **Choose Setup:** zero-shot / one-shot / few-shot (see Section 4.2 of the paper)
   - For example, use `--one_shot` for one-shot, or `--demo_consistent` for 8-shot consistent demonstrations.
3. **Run Model:** Select an LLM with multilingual capabilities.

#### Example Command-Line Usage

**8-shot (consistent) in-context learning:**
```bash
python icl.py --lang1 en --lang2 de --testdata BMIKE53/zsRE/zsre_test_ --manualdata zsre_multi --model_name meta-llama/Meta-Llama-3.1-8B --demo_consistent
```

**One-shot in-context learning:**
```bash
python icl.py --lang1 en --lang2 zh --testdata BMIKE53/zsRE/zsre_test_ --manualdata zsre_multi --model_name meta-llama/Meta-Llama-3.2-3B --one_shot
```

- Replace `$LANG` with the target language code (e.g., `de` for German, `fr` for French, etc.).
- The `--testdata` argument should point to the prefix of the test file (e.g., `BMIKE53/zsRE/zsre_test_`), and the script will append the language code automatically.
- The `--manualdata` argument specifies the demonstration pool.
- The `--model_name` argument selects the model checkpoint to use.

---

## Citation

If you use BMIKE-53, please cite:

```bibtex
@inproceedings{nie2025bmike53,
  title={BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning},
  author={Nie, Ercong and Shao, Bo and Wang, Mingyang and Ding, Zifeng and Schmid, Helmut and Sch{\"u}tze, Hinrich},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
