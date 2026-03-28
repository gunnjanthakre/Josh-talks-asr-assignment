# Josh Talks ASR Assignment

This repository contains code, data, and results for the ASR assignment completed for Josh Talks. The work covers:

- **Fine-tuning Whisper-small** on Josh Talks Hindi data and evaluating on FLEURS Hindi test set.
- **ASR cleanup pipeline** for number normalization and English word detection.
- **Spell checking** of ~177,500 unique Hindi words with confidence scoring.
- **Lattice-based WER evaluation** for fairer ASR model comparison.

The full report with detailed analysis is available in the [Assignment Document](https://docs.google.com/document/d/10VCxvDeZztrbFK9GHqCuYLUohGQtex0MSDgQFu4b4nI/edit?usp=sharing). *(Replace with your actual link or remove if not public.)*

## 📁 Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `Untitled0 (1).ipynb` | Main notebook for Q1: data preprocessing, feature extraction, and Whisper fine-training. |
| `features/` | *(not tracked due to size)* Contains `train.pkl` and `eval.pkl` with pre-processed features. |
| `whisper-small-hindi/` | Fine-tuned model (saved after training). |
| `fleurs_results.csv` | Predictions on FLEURS Hindi test set (baseline vs. fine-tuned). |
| `error_samples.csv` | 25 sampled error utterances for error analysis. |
| `q2_pipeline.py` | Cleanup pipeline: number normalization + English word tagging. |
| `q2_pipeline_results.xlsx` | Output of the cleanup pipeline on the full dataset. |
| `q3_spell_checker.py` | Spell checking for unique words. |
| `q3_word_classifications.xlsx` | Final list of unique words with spelling status. |
| `q3_full_results.xlsx` | Full results including confidence and reason. |
| `q3_low_confidence_review.xlsx` | 50 low-confidence words for manual inspection. |
| `q4_lattice_wer.py` | Lattice-based WER evaluation. |
| `q4_results.xlsx` | Lattice WER results for all models. |
| `Question 4.xlsx` | Input data for lattice evaluation. |
| `Unique Words Data.xlsx` | Input unique words for spell checking. |
| `FT Result.xlsx` | *(Original metadata file – reference only)* |
| `corpus_texts.json` | Extracted texts from the dataset (used for building dictionary). |

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install torch transformers datasets accelerate evaluate jiwer librosa soundfile pandas openpyxl tqdm huggingface_hub
```

**A GPU is highly recommended** for training and evaluation.

## 🚀 Quick Start

### Question 1 – Fine-tuning Whisper-small

**Preprocessing & Training** (in `Untitled0 (1).ipynb`):
1. Place `FT Data.xlsx` in the repository root.
2. Run the notebook cells sequentially.

*Timeline*: 
- Audio downloading + feature extraction: ~20 minutes
- Training: ~30-40 minutes on T4 GPU

**Results**:
- Fine-tuned model saved in `whisper-small-hindi/`
- **Baseline WER**: 56.4%
- **Fine-tuned WER**: 54.9% (FLEURS Hindi test set)

### Question 2 – ASR Cleanup Pipeline

```bash
python q2_pipeline.py
```

**Features**:
- Number normalization (Hindi numbers → digits)
- English word detection & tagging `[EN]...[/EN]`

**Example transformations**:
"तीन सौ चौवन लोग आए" → "354 लोग आए"
"दो-चार बातें करनी थीं" → unchanged (idiom protected)
"मेरा इंटरव्यू अच्छा गया" → "मेरा [EN]इंटरव्यू[/EN] अच्छा गया"


**Output**: `q2_pipeline_results.xlsx`

### Question 3 – Spell Checking Unique Words

```bash
python q3_spell_checker.py
```

**Input**: `Unique Words Data.xlsx` (177,509 unique words)

**Results**:
- **Correctly spelled**: 162,324 (91.4%)
- **Incorrectly spelled**: 15,185 (8.6%)

**Outputs**:
- `q3_word_classifications.xlsx` – word + spelling status
- `q3_full_results.xlsx` – confidence + reason
- `q3_low_confidence_review.xlsx` – 50 words for manual review

### Question 4 – Lattice-based WER Evaluation

```bash
python q4_lattice_wer.py
```

**Method**:
- Builds lattice with reference variants + model consensus (≥3 models agree)
- Computes standard WER vs. lattice WER

**Sample Results** (`q4_results.xlsx`):

| Model | Standard WER | Lattice WER | Improvement (Δ) | Unfairly Penalized |
|-------|--------------|-------------|-----------------|-------------------|
| Model H | 0.0331 | 0.0290 | 0.0041 | 4 |
| Model i | 0.0061 | 0.0061 | 0.0000 | 0 |
| Model k | 0.1060 | 0.0746 | 0.0314 | 17 |

## 📝 Notes

- Large files (`train.pkl`, `eval.pkl`, `whisper-small-hindi/`) excluded due to size limits.
- `FT Data.xlsx` not included for privacy (expected in root for notebook).
- All result files (Excel/CSV) are included for inspection.

## 🔗 Links
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- [Whisper Model](https://huggingface.co/openai/whisper-small)

---
*Built with ❤️ for Josh Talks ASR Assignment*
