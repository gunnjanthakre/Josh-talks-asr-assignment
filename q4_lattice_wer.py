"""
Q4: Lattice-Based WER Evaluation (FINAL CORRECT VERSION)
=========================================================
THEORY:
Standard WER compares model output against ONE rigid reference string.
Problem: Hindi speech has valid variants — punctuation differences,
spelling variants (खेती बाड़ी vs खेतीबाड़ी), number forms (14 vs चौदह).

Solution: Build a LATTICE — a sequential list of "bins" where each bin
holds ALL valid alternatives for that position. A model is correct if
it matches ANY entry in its bin.

IMPORTANT: Lattice WER must always be <= Standard WER.
The lattice can only HELP models, never hurt them.

ALIGNMENT UNIT: WORD-level
Justification: ASR output is naturally word-tokenized. Word-level WER
is the industry standard for Hindi ASR evaluation.
"""

import pandas as pd
import re
from collections import Counter

# ─── 1. Load real data ────────────────────────────────────────────────────────
df = pd.read_excel("/Users/gunjanthakre/Downloads/josh_talks_asr/Question 4.xlsx")
df = df.dropna(subset=["Human"])
model_cols = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]

# ─── 2. Normalization ─────────────────────────────────────────────────────────
def normalize(text):
    """Remove punctuation, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[।,\.!?\-–—]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── 3. Valid variant check ───────────────────────────────────────────────────
def are_valid_variants(ref_word, hyp_word):
    """
    True if hyp_word is an acceptable alternative to ref_word.
    Covers: punctuation variants, compound splitting, 1-char spelling diff.
    """
    r = normalize(ref_word)
    h = normalize(hyp_word)
    if r == h:
        return True
    # Compound word split/merge: खेतीबाड़ी <-> खेती + बाड़ी
    if r in h or h in r:
        return True
    # Single character difference (minor spelling variant)
    if abs(len(r) - len(h)) == 0 and len(r) >= 3:
        diffs = sum(1 for a, b in zip(r, h) if a != b)
        if diffs <= 1:
            return True
    if abs(len(r) - len(h)) == 1 and len(r) >= 4:
        # One char inserted/deleted
        shorter, longer = (r, h) if len(r) < len(h) else (h, r)
        for i in range(len(longer)):
            candidate = longer[:i] + longer[i+1:]
            if candidate == shorter:
                return True
    return False

# ─── 4. Standard edit-distance WER ───────────────────────────────────────────
def edit_distance(ref, hyp):
    """Standard word-level edit distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def compute_wer(reference, hypothesis):
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)

# ─── 5. Build Lattice ─────────────────────────────────────────────────────────
def build_lattice(reference, all_model_outputs, majority_threshold=3):
    """
    For each word position in the reference, collect valid alternatives.

    RULE 1: Always include the reference word itself.
    RULE 2: Add a model's word if it's a valid variant of the reference word
            (punctuation, compound split, 1-char spelling difference).
    RULE 3: If >= majority_threshold models agree on a word at a position,
            add it — this handles cases where the reference itself is wrong.
    """
    ref_tokens = reference.split()
    n = len(ref_tokens)
    # Start with reference words
    lattice = [set([tok]) for tok in ref_tokens]

    # For majority voting: what does each model say at each position?
    position_votes = [[] for _ in range(n)]

    for model_out in all_model_outputs:
        if not model_out:
            continue
        hyp_tokens = model_out.split()
        # Simple position-by-position comparison (min length)
        for i in range(min(n, len(hyp_tokens))):
            hyp_word = hyp_tokens[i]
            ref_word = ref_tokens[i]
            # Rule 2: valid variant
            if are_valid_variants(ref_word, hyp_word):
                lattice[i].add(hyp_word)
            # Always record for majority vote
            position_votes[i].append(hyp_word)

    # Rule 3: majority vote
    for i in range(n):
        if not position_votes[i]:
            continue
        most_common, count = Counter(position_votes[i]).most_common(1)[0]
        if count >= majority_threshold:
            lattice[i].add(most_common)

    return lattice

# ─── 6. Lattice WER ──────────────────────────────────────────────────────────
def compute_lattice_wer_dp(lattice, hypothesis):
    """
    Proper lattice-based WER using dynamic programming.
    
    Key insight: we replace the reference string with a lattice.
    At each position, a word matches if it's in the lattice bin.
    We use the same edit-distance DP but with set-membership check.
    
    This GUARANTEES lattice WER <= standard WER because:
    - The lattice always contains the reference word
    - Any additional alternatives can only reduce errors, never increase them
    """
    hyp_tokens = hypothesis.split()
    n = len(lattice)  # reference length
    m = len(hyp_tokens)

    if n == 0:
        return 0.0

    # DP: dp[i][j] = min edits to match lattice[:i] with hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i  # deletions
    for j in range(m + 1): dp[0][j] = j  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Check if hyp word matches ANY alternative in lattice bin i-1
            if hyp_tokens[j-1] in lattice[i-1]:
                dp[i][j] = dp[i-1][j-1]  # match — no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[n][m] / n

# ─── 7. Run on all rows ───────────────────────────────────────────────────────
results = {col: {"std_wer": [], "lat_wer": []} for col in model_cols}
lattice_examples = []

for idx, row in df.iterrows():
    ref = normalize(str(row["Human"]))
    if not ref:
        continue

    model_outputs = [
        normalize(str(row[c])) if pd.notna(row[c]) else ""
        for c in model_cols
    ]

    lattice = build_lattice(ref, model_outputs)

    if idx < 3:
        lattice_examples.append({
            "row": idx,
            "reference": ref,
            "lattice": [sorted(s) for s in lattice]
        })

    for col, model_out in zip(model_cols, model_outputs):
        if not model_out:
            continue
        std = compute_wer(ref, model_out)
        lat = compute_lattice_wer_dp(lattice, model_out)
        results[col]["std_wer"].append(std)
        results[col]["lat_wer"].append(lat)

# ─── 8. Print Results ─────────────────────────────────────────────────────────
print("=" * 70)
print("LATTICE EXAMPLES (first 3 utterances)")
print("=" * 70)
for ex in lattice_examples:
    print(f"\nRow {ex['row']} | Reference: {ex['reference']}")
    for i, alts in enumerate(ex["lattice"]):
        if len(alts) > 1:
            print(f"  Position {i}: {alts}  ← valid alternatives")
        else:
            print(f"  Position {i}: {alts}")

print("\n" + "=" * 70)
print(f"{'Model':<10} {'Std WER':>10} {'Lattice WER':>13} {'Δ improvement':>15} {'Notes'}")
print("-" * 70)

summary_rows = []
for col in model_cols:
    s = results[col]["std_wer"]
    l = results[col]["lat_wer"]
    if not s:
        continue
    avg_std = sum(s) / len(s)
    avg_lat = sum(l) / len(l)
    delta = avg_std - avg_lat
    unfair = sum(1 for a, b in zip(s, l) if a > b)

    if delta > 0.001:
        note = f"✅ unfairly penalized in {unfair} utterances"
    elif delta == 0:
        note = "— errors are genuine"
    else:
        note = "— check data"

    print(f"{col:<10} {avg_std:>10.3f} {avg_lat:>13.3f} {delta:>+15.3f}  {note}")
    summary_rows.append({
        "Model": col,
        "Avg Standard WER": round(avg_std, 4),
        "Avg Lattice WER": round(avg_lat, 4),
        "Improvement (Δ)": round(delta, 4),
        "Utterances Unfairly Penalized": unfair,
        "Total Utterances Evaluated": len(s)
    })

print("=" * 70)
print("""
INTERPRETATION:
  Positive Δ  → model was unfairly penalized by rigid single reference
  Zero Δ      → model's errors are genuine mistakes (lattice cannot help)
  Lattice WER is always <= Standard WER (mathematically guaranteed)

WHAT THE LATTICE CAPTURES FROM THIS DATA:
  • Punctuation variants:  है। vs है  (Model i adds । — valid!)
  • Compound splitting:    खेतीबाड़ी vs खेती बाड़ी  (both correct spellings)
  • Spelling variants:     मौनता vs मोनता  (1-char difference, dialectal)
  • Majority agreement:    if 3+ models agree, reference may be wrong

HOW WE DECIDE TO TRUST MODELS OVER REFERENCE:
  When >= 3 of 5 models produce the same word at a position, we add
  it to the lattice. This handles human transcription errors where
  the reference itself is incorrect.
""")

out_df = pd.DataFrame(summary_rows)
out_df.to_excel("q4_results.xlsx", index=False)
print("✅ Results saved to q4_results.xlsx")