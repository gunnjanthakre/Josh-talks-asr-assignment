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


pip install torch transformers datasets accelerate evaluate jiwer librosa soundfile pandas openpyxl tqdm huggingface_hub
A GPU is highly recommended for training and evaluation.

Question 1 – Fine‑tuning Whisper‑small
Preprocessing
The preprocessing steps are implemented in Untitled0 (1).ipynb:

Extracts folder_id from rec_url_gcp and constructs the correct transcription URL.

Downloads and segments audio, keeping segments between 1 and 30 seconds.

Resamples to 16 kHz and extracts log‑mel spectrograms.

Tokenizes ground‑truth text.

Splits by recording_id (80% train, 20% eval).

Training
The notebook also contains the training loop with mixed precision (AMP). To run:

Place FT Data.xlsx in the repository root.

Run the notebook cells sequentially.
Note: The audio downloading and feature extraction step may take ~20 minutes. The training itself takes ~30‑40 minutes on a T4 GPU.

The fine‑tuned model is saved in whisper-small-hindi/.

Evaluation on FLEURS
The evaluation is performed in the same notebook (towards the end). It loads the baseline and fine‑tuned models, runs inference on 100 FLEURS Hindi test samples, and saves results in fleurs_results.csv.

Results:

Baseline WER: 56.4%

Fine‑tuned WER: 54.9%

Error samples (25) are saved in error_samples.csv.

Question 2 – ASR Cleanup Pipeline
The pipeline is in q2_pipeline.py. To run:

bash
python q2_pipeline.py
This will:

Download the dataset from the provided URL (or use a fallback).

Apply number normalization (converts Hindi number words to digits, handling idioms and sequences).

Detect English words (using a predefined list) and tag them with [EN]...[/EN].

Save the results in q2_pipeline_results.xlsx.

The output file contains:

Original text

Text after number normalization

Text after English tagging

Count of number conversions

List of English words found

Example transformations
"तीन सौ चौवन लोग आए" → "354 लोग आए"

"दो-चार बातें करनी थीं" → unchanged (idiom protected)

"मेरा इंटरव्यू अच्छा गया" → "मेरा [EN]इंटरव्यू[/EN] अच्छा गया"

Question 3 – Spell Checking Unique Words
Run the spell checker:

bash
python q3_spell_checker.py
This loads Unique Words Data.xlsx (177,509 unique words) and classifies each word as 'correct spelling' or 'incorrect spelling' with a confidence level (high/medium/low). Outputs:

q3_word_classifications.xlsx – two columns: word and spelling status.

q3_full_results.xlsx – includes confidence and reason.

q3_low_confidence_review.xlsx – first 50 low‑confidence words for manual review.

Results:

Correctly spelled: 162,324 (91.4%)

Incorrectly spelled: 15,185 (8.6%)

Question 4 – Lattice‑Based WER Evaluation
Run the lattice evaluation:

bash
python q4_lattice_wer.py
This reads Question 4.xlsx (which contains human reference and 6 model outputs) and computes both standard WER and lattice‑based WER for each model. The lattice is built by:

Including valid variants of reference words (punctuation, compound splits, 1‑char differences).

Adding words on which ≥3 models agree (to account for possible reference errors).

Results are saved in q4_results.xlsx and q4_results.csv.

Sample output (from the actual run):

Model	Standard WER	Lattice WER	Improvement (Δ)	Unfairly Penalized Utterances
Model H	0.0331	0.0290	0.0041	4
Model i	0.0061	0.0061	0.0000	0
Model k	0.1060	0.0746	0.0314	17
Model l	0.1066	0.0636	0.0430	23
Model m	0.2012	0.1113	0.0899	33
Model n	0.1073	0.0715	0.0358	22
The lattice method reduces WER for models that were unfairly penalized by a rigid reference, while leaving it unchanged for models whose errors are genuine.

Notes
The large pre‑processed feature files (train.pkl, eval.pkl) and the fine‑tuned model (whisper-small-hindi/) are not included in the repository due to size limits. They can be recreated by running the notebook.

The original FT Data.xlsx is also not included for privacy; the notebook expects it in the root directory.

All results in Excel/CSV files are included for inspection.
