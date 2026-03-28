# Josh Talks ASR Assignment

This repository contains code, data, and results for the ASR assignment completed for Josh Talks. The work covers:

- **Fine-tuning Whisper‑small** on Josh Talks Hindi data and evaluating on FLEURS Hindi test set.
- **ASR cleanup pipeline** for number normalization and English word detection.
- **Spell checking** of ~177,500 unique Hindi words with confidence scoring.
- **Lattice‑based WER evaluation** for fairer ASR model comparison.

The full report with detailed analysis is available in the [Assignment Document](https://docs.google.com/document/d/...). (Replace with your actual link or remove if not public.)

---

## Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `Untitled0 (1).ipynb` | Main notebook for Q1: data preprocessing, feature extraction, and Whisper fine‑training. |
| `features/` | (not tracked due to size) Contains `train.pkl` and `eval.pkl` with pre‑processed features. |
| `whisper-small-hindi/` | Fine‑tuned model (saved after training). |
| `fleurs_results.csv` | Predictions on FLEURS Hindi test set (baseline vs. fine‑tuned). |
| `error_samples.csv` | 25 sampled error utterances for error analysis. |
| `q2_pipeline.py` | Cleanup pipeline: number normalization + English word tagging. |
| `q2_pipeline_results.xlsx` | Output of the cleanup pipeline on the full dataset. |
| `q3_spell_checker.py` | Spell checking for unique words. |
| `q3_word_classifications.xlsx` | Final list of unique words with spelling status. |
| `q3_full_results.xlsx` | Full results including confidence and reason. |
| `q3_low_confidence_review.xlsx` | 50 low‑confidence words for manual inspection. |
| `q4_lattice_wer.py` | Lattice‑based WER evaluation. |
| `q4_results.xlsx` | Lattice WER results for all models. |
| `Question 4.xlsx` | Input data for lattice evaluation. |
| `Unique Words Data.xlsx` | Input unique words for spell checking. |
| `FT Result.xlsx` | (Original metadata file – not used directly in code, but reference.) |
| `corpus_texts.json` | Extracted texts from the dataset (used for building dictionary). |

---

## Requirements

Install the required Python packages:

```bash
pip install torch transformers datasets accelerate evaluate jiwer librosa soundfile pandas openpyxl tqdm huggingface_hub
