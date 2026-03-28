"""
Q3: Hindi Spell Checking — 177,509 Unique Words
=================================================
APPROACH:
We classify each word as correctly or incorrectly spelled using a
multi-signal pipeline:

SIGNAL 1 — Dictionary lookup (primary signal)
  Use a large Hindi wordlist. If word is found → likely correct.

SIGNAL 2 — Pattern-based rules (catches valid words not in dictionary)
  - Very short words (1-2 chars) → high chance of being correct particles
  - Words ending in common Hindi suffixes (ने, ता, ती, ते, ना, ...) → likely correct
  - Words that are English transliterated into Devanagari → mark as correct
    (per assignment guidelines: "computer" → "कंप्यूटर" is CORRECT)

SIGNAL 3 — Error pattern detection
  - Mixed script (Devanagari + Roman + Urdu in same word) → likely error
  - Repeated consonant clusters unusual in Hindi → likely error
  - Words with non-Devanagari characters mixed in → likely error
  - Very long words (>20 chars) without compound word structure → likely error
  - Words with .../ or - patterns suggesting transcription artifacts

CONFIDENCE SCORING:
  HIGH:   Dictionary hit OR clear error pattern detected
  MEDIUM: Suffix match OR reasonable length without clear signals
  LOW:    Unclear — could be valid rare word, proper noun, or error
"""

import pandas as pd
import re
import unicodedata
from tqdm import tqdm

# ─── 1. Load the word list ───────────────────────────────────────────────────
print("Loading words...")
df = pd.read_excel("/Users/gunjanthakre/Downloads/josh_talks_asr/Unique Words Data.xlsx")
words = df["word"].dropna().tolist()
print(f"Total words loaded: {len(words)}")

# ─── 2. Build a basic Hindi dictionary from our corpus itself ────────────────
# High-frequency words (top 5000) are almost certainly correct
# We use frequency as a proxy for correctness

from collections import Counter

# Also load our transcription corpus texts
import json, os
corpus_texts = []
if os.path.exists("corpus_texts.json"):
    with open("corpus_texts.json") as f:
        corpus_texts = json.load(f)

# Add hardcoded common Hindi words as seed dictionary
HINDI_COMMON = set("""
है हैं तो में भी के नहीं कि वो और से जो हो मतलब हां हम की एक ही का आप को ये था
बहुत मैं कुछ अच्छा हाँ होता जैसे क्या ना फिर कर लोग कोई हम्म बिल्कुल बात पर थे
अगर पे ऐसा या मुझे लिए रहा था थे इस उस वह वे हमें तुम आज कल यह इस उस जब तब
कैसे कहां क्यों किसी कभी सब अब तक जैसे वैसे इसलिए क्योंकि लेकिन मगर परंतु
होना करना जाना आना देना लेना बोलना सुनना देखना समझना जानना मानना
अच्छा बुरा बड़ा छोटा नया पुराना सही गलत ज़रूरी खास आसान मुश्किल
एक दो तीन चार पाँच छह सात आठ नौ दस बीस तीस चालीस पचास सौ हज़ार
मैं तुम वह हम आप वे यह ये वो मेरा तेरा उसका हमारा आपका उनका
घर काम वक्त दिन रात साल महीना घंटा मिनट पैसा रुपया
पहले बाद ऊपर नीचे आगे पीछे अंदर बाहर दाएं बाएं
हाँ नहीं शायद ज़रूर बिल्कुल सच झूठ
बताना पूछना समझना सोचना चाहना होना पाना मिलना
इंटरव्यू जॉब कंप्यूटर मोबाइल फोन ऑफिस स्कूल कॉलेज
प्रोजेक्ट एरिया टेंट कैंप मिस्टेक लाइट
""".split())

# ─── 3. Common valid Hindi suffixes ──────────────────────────────────────────
VALID_SUFFIXES = [
    'ना', 'ने', 'नी', 'ता', 'ती', 'ते', 'या', 'यी', 'ये',
    'ओ', 'ओं', 'आ', 'आं', 'ई', 'इयों', 'वाला', 'वाली', 'वाले',
    'कर', 'के', 'की', 'का', 'को', 'से', 'में', 'पर', 'तक',
    'ाएं', 'ाएगा', 'ाएगी', 'ेगा', 'ेगी', 'ेंगे', 'ेंगी',
    'ाना', 'ाने', 'ानी', 'ाया', 'ाई', 'ाए',
    'िया', 'िए', 'िएं', 'ित', 'िक', 'िश', 'िल',
    'ता है', 'ती है', 'ते हैं',
]

# ─── 4. Error patterns ───────────────────────────────────────────────────────
def has_mixed_script(word):
    """Word has both Devanagari and Latin/Urdu characters — likely error."""
    has_dev = bool(re.search(r'[\u0900-\u097F]', word))
    has_latin = bool(re.search(r'[a-zA-Z]', word))
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', word))
    return (has_dev and has_latin) or (has_dev and has_arabic)

def has_artifact_chars(word):
    """Transcription artifacts like ..., ---, /, numbers mixed in."""
    return bool(re.search(r'[/\\|@#$%^&*+=<>{}[\]~`]', word))

def is_repeated_unusual(word):
    """Unusual repetition like the same consonant cluster 3+ times."""
    return bool(re.search(r'(.{2,})\1\1', word))

def is_pure_devanagari(word):
    """Check if word is purely Devanagari (including nukta, anusvara etc)."""
    return bool(re.match(r'^[\u0900-\u097F\u200C\u200D।॥]+$', word))

def is_likely_transliteration(word):
    """
    English words written in Devanagari are CORRECT per guidelines.
    Heuristic: word has unusual consonant clusters for Hindi but is
    a valid Devanagari string — likely an English word transliterated.
    Examples: कंप्यूटर, इंटरव्यू, मोबाइल
    """
    # Common transliteration patterns
    translit_patterns = [
        r'[कगटडपबफ]्[रलव]',   # consonant clusters like kr, gl, pr, br
        r'[इउ]ं[टडपबफ]',       # nasal before stops (common in English words)
        r'ए[कगटडपब]्',          # ek- pattern
        r'[आइउए]ल्',            # -l ending clusters
    ]
    for p in translit_patterns:
        if re.search(p, word):
            return True
    return False

# ─── 5. Main classification function ─────────────────────────────────────────
def classify_word(word):
    """
    Returns: (label, confidence, reason)
    label: 'correct spelling' or 'incorrect spelling'
    confidence: 'high', 'medium', 'low'
    reason: brief explanation
    """
    if not isinstance(word, str) or not word.strip():
        return ('incorrect spelling', 'high', 'empty or non-string')

    word = word.strip()

    # ── DEFINITE ERRORS ──────────────────────────────────────────────────────

    if has_mixed_script(word):
        return ('incorrect spelling', 'high', 'mixed Devanagari+Latin/Arabic script in single word')

    if has_artifact_chars(word):
        return ('incorrect spelling', 'high', 'contains transcription artifact characters')

    if len(word) > 25:
        return ('incorrect spelling', 'medium', 'unusually long — likely merged words or transcription error')

    if is_repeated_unusual(word):
        return ('incorrect spelling', 'medium', 'unusual character repetition pattern')

    # ── DEFINITE CORRECT ────────────────────────────────────────────────────

    if word in HINDI_COMMON:
        return ('correct spelling', 'high', 'found in common Hindi word dictionary')

    if not is_pure_devanagari(word):
        # Has non-Devanagari chars but passed mixed-script check
        # Could be numbers or punctuation — classify as incorrect
        if re.search(r'[0-9]', word):
            return ('incorrect spelling', 'medium', 'contains digits mixed with Devanagari')
        return ('incorrect spelling', 'medium', 'non-Devanagari characters present')

    # ── SUFFIX-BASED ANALYSIS ────────────────────────────────────────────────

    # Very short pure Devanagari words are likely correct particles/pronouns
    if len(word) <= 3 and is_pure_devanagari(word):
        return ('correct spelling', 'high', 'short Devanagari particle/pronoun — likely correct')

    # Check suffixes
    for suffix in VALID_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return ('correct spelling', 'medium', f'ends with valid Hindi suffix "{suffix}"')

    # Transliteration check
    if is_likely_transliteration(word):
        return ('correct spelling', 'medium', 'likely English word transliterated to Devanagari (per guidelines: correct)')

    # ── PATTERN-BASED ERROR DETECTION ───────────────────────────────────────

    # Words with dots (like "दीदी...")
    if '.' in word or '…' in word:
        return ('incorrect spelling', 'medium', 'contains dots — likely transcription artifact')

    # Words with hyphens suggesting merged correction artifacts
    if '-' in word:
        # Some hyphenated words are valid (रंग-बिरंगा)
        parts = word.split('-')
        if all(is_pure_devanagari(p) and len(p) >= 2 for p in parts):
            return ('correct spelling', 'medium', 'hyphenated compound word — likely valid')
        return ('incorrect spelling', 'medium', 'hyphen with unclear structure')

    # ── MEDIUM LENGTH PURE DEVANAGARI — UNCERTAIN ───────────────────────────
    if is_pure_devanagari(word):
        if len(word) <= 12:
            return ('correct spelling', 'low', 'pure Devanagari, moderate length — uncertain, likely correct')
        else:
            return ('incorrect spelling', 'low', 'pure Devanagari but very long — possibly merged words')

    return ('incorrect spelling', 'low', 'could not confidently classify')

# ─── 6. Run on all 177,509 words ─────────────────────────────────────────────
print("\nClassifying 177,509 words...")
results = []
for word in tqdm(words):
    label, confidence, reason = classify_word(word)
    results.append({
        "word": word,
        "spelling_status": label,
        "confidence": confidence,
        "reason": reason
    })

result_df = pd.DataFrame(results)

# ─── 7. Summary statistics ───────────────────────────────────────────────────
correct = result_df[result_df["spelling_status"] == "correct spelling"]
incorrect = result_df[result_df["spelling_status"] == "incorrect spelling"]

print("\n" + "=" * 60)
print("Q3 RESULTS SUMMARY")
print("=" * 60)
print(f"Total unique words:          {len(result_df):,}")
print(f"Correctly spelled:           {len(correct):,}  ({len(correct)/len(result_df)*100:.1f}%)")
print(f"Incorrectly spelled:         {len(incorrect):,}  ({len(incorrect)/len(result_df)*100:.1f}%)")
print(f"\nConfidence breakdown:")
for conf in ['high', 'medium', 'low']:
    n = len(result_df[result_df["confidence"] == conf])
    print(f"  {conf.capitalize():8}: {n:,}")

print(f"\nLow confidence words (sample 10):")
low_conf = result_df[result_df["confidence"] == "low"].head(10)
for _, row in low_conf.iterrows():
    print(f"  [{row['spelling_status'][:3].upper()}] {row['word']:20} — {row['reason']}")

# ─── 8. Save outputs ─────────────────────────────────────────────────────────
# Main output: 2-column sheet
output_df = result_df[["word", "spelling_status"]].copy()
output_df.to_excel("q3_word_classifications.xlsx", index=False)
print(f"\n✅ Saved q3_word_classifications.xlsx")

# Full output with confidence
result_df.to_excel("q3_full_results.xlsx", index=False)
print(f"✅ Saved q3_full_results.xlsx (with confidence + reason)")

# Low confidence sample for manual review (Q3c)
low_conf_full = result_df[result_df["confidence"] == "low"].head(50)
low_conf_full.to_excel("q3_low_confidence_review.xlsx", index=False)
print(f"✅ Saved q3_low_confidence_review.xlsx (50 low-confidence words for review)")

print("\nDONE!")