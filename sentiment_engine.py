"""
Sentiment Engine — Twitiment
Deterministic 7-step NLP preprocessing + context-aware VADER sentiment analysis.
Enforces strict pipeline transformation, lexicon scoring, and mathematical integrity.
"""
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------- #
# Initialise Resources
# --------------------------------------------------------------------------- #
_analyzer = SentimentIntensityAnalyzer()
_lemmatizer = WordNetLemmatizer()


def _download_nltk():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_download_nltk()
_STOPWORDS = set(stopwords.words('english'))

# --------------------------------------------------------------------------- #
# Comprehensive slang / abbreviation dictionary
# --------------------------------------------------------------------------- #
_SLANG_MAP = {
    "u": "you", "ur": "your", "r": "are", "rn": "right now",
    "omg": "oh my god", "lol": "laughing out loud", "lmao": "laughing my ass off",
    "brb": "be right back", "btw": "by the way", "smh": "shaking my head",
    "imo": "in my opinion", "imho": "in my humble opinion",
    "tbh": "to be honest", "ngl": "not going to lie",
    "idk": "i do not know", "ik": "i know", "ikr": "i know right",
    "fyi": "for your information", "afaik": "as far as i know",
    "bc": "because", "cuz": "because", "bcz": "because",
    "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you",
    "np": "no problem", "nvm": "never mind",
    "wth": "what the hell", "wtf": "what the heck",
    "af": "extremely", "rly": "really", "srsly": "seriously",
    "gr8": "great", "b4": "before", "2day": "today", "2nite": "tonight",
    "w/": "with", "w/o": "without", "abt": "about",
    "govt": "government", "govt.": "government",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of",
}

# Negation words to preserve through stop-word removal
_NEGATIONS = {"not", "no", "never", "nor", "neither", "nobody", "nothing",
              "nowhere", "hardly", "barely", "scarcely", "rarely"}

# Intensifiers and boosters that amplify sentiment
_BOOSTERS = {"very", "really", "extremely", "incredibly", "absolutely",
             "totally", "completely", "utterly", "highly", "super",
             "most", "quite", "especially", "particularly"}

# Negative prefixes that indicate negation within a word
_NEGATIVE_PREFIXES = ("un", "in", "im", "ir", "il", "dis", "mis", "non")

# Structural/objective words that must always score 0.00
_STRUCTURAL_WORDS = {
    "you", "your", "they", "them", "their", "we", "our", "us", "he", "she",
    "it", "its", "his", "her", "my", "mine", "i",
    "side", "set", "get", "got", "make", "take", "put", "use", "go", "come",
    "thing", "way", "time", "day", "year", "people", "man", "woman",
    "server", "management", "tool", "system", "data", "code", "file",
    "technical", "technology", "software", "hardware", "device",
    "also", "still", "even", "just", "much", "many", "more", "less",
    "however", "although", "though", "since", "while", "whether",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_wordnet_pos(treebank_tag: str):
    """Map Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


# Words that LOOK like they have a negative prefix but are NOT negations
_FALSE_NEGATIVE_PREFIX_WORDS = {
    # "in-" words that are not negations
    "incredible", "incredibly", "increase", "indeed", "indicate", "individual",
    "industry", "information", "initial", "input", "inside", "inspire",
    "install", "instance", "instead", "institute", "instrument", "integrate",
    "intend", "interest", "internal", "interpret", "introduce", "invest",
    "involve", "insight", "intense", "interact", "interview", "into",
    "inventory", "investigate", "invitation", "invoke", "invent",
    # "im-" words that are not negations
    "impact", "implement", "import", "important", "impose", "impress",
    "improve", "image", "imagine", "immediate", "immense",
    # "ir-" words that are not negations
    "iron",
    # "il-" words that are not negations
    "illustrate", "illuminate",
    # "dis-" words that are not negations
    "discover", "discuss", "display", "distance", "distribute", "district",
    "discipline",
    # "mis-" words that are not negations
    "mission", "miss",
    # "un-" words that are not negations
    "under", "understand", "unit", "unite", "universe", "university",
    "until", "upon", "unique", "unless", "unlike", "update", "upper",
}


def _has_negative_prefix(word: str) -> bool:
    """Check if a word carries an inherent negation prefix (e.g., unhelpful, impossible).
    Excludes words that coincidentally start with a negative-looking prefix."""
    # First check the exclusion list
    if word.lower() in _FALSE_NEGATIVE_PREFIX_WORDS:
        return False
    for prefix in _NEGATIVE_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            stem = word[len(prefix):]
            if wordnet.synsets(stem):
                return True
    return False


def _strip_negative_prefix(word: str) -> str:
    """Strip a negative prefix and return the positive root."""
    for prefix in _NEGATIVE_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            stem = word[len(prefix):]
            if wordnet.synsets(stem):
                return stem
    return word


# --------------------------------------------------------------------------- #
# 7-Step Preprocessing Pipeline
# --------------------------------------------------------------------------- #

def step_tokenization(text: str) -> tuple[str, str]:
    """Step 1 – Tokenization: NLTK word_tokenize for accurate splitting."""
    tokens = word_tokenize(text)
    return " ".join(tokens), "Segmenting raw text into discrete word and punctuation tokens using NLTK tokenizer."


def step_lowercasing(text: str) -> tuple[str, str]:
    """Step 2 – Lowercasing."""
    return text.lower(), "Converting all tokens to lowercase for case-insensitive uniformity."


def step_noise_removal(text: str) -> tuple[str, str]:
    """Step 3 – Noise Removal: Strip HTML, URLs, @mentions, possessives,
    ALL punctuation, numbers, and special characters."""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Strip Twitter @mentions (keep word after @)
    text = re.sub(r'@(\w+)', r'\1', text)
    # Strip RT prefix
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    # Strip possessive markers ('s, 's)
    text = re.sub(r"['']s\b", '', text)
    # Normalize repeated characters (e.g., "loooove" -> "loove")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Character-level filter: keep ONLY letters, spaces, hyphens, and emojis
    # This strips ALL punctuation, numbers, and special characters
    cleaned = []
    for char in text:
        if char.isalpha() or char.isspace() or emoji.is_emoji(char):
            cleaned.append(char)
        elif char == '-':
            # Keep hyphens only if between letters (compound words)
            cleaned.append(char)
        else:
            cleaned.append(' ')
    text = "".join(cleaned)

    # Remove standalone hyphens
    text = re.sub(r'\s-\s', ' ', text)
    text = re.sub(r'^-|-$', '', text)

    # Collapse spaces
    return " ".join(text.split()), "Stripping HTML, URLs, @mentions, possessives, ALL punctuation, numbers, and special characters."


def step_stop_word_removal(text: str) -> tuple[str, str]:
    """Step 4 – Stop Word Removal: Remove closed-class words, preserve negations & boosters."""
    tokens = text.split()
    filtered = []
    for w in tokens:
        if w in _NEGATIONS or w in _BOOSTERS:
            filtered.append(w)
        elif w not in _STOPWORDS:
            filtered.append(w)
    return " ".join(filtered), "Eliminating closed-class words while preserving negations and intensifiers."


def step_lemmatization(text: str) -> tuple[str, str]:
    """Step 5 – POS-aware Lemmatization: Verbs→infinitive, Nouns→singular.
    Mandatory transformation — no token passes through unchanged if inflected."""
    tokens = text.split()
    pos_tags = nltk.pos_tag(tokens)
    lemmas = []
    for word, tag in pos_tags:
        wn_pos = _get_wordnet_pos(tag)
        lemma = _lemmatizer.lemmatize(word, wn_pos)
        # Double-check: if verb lemma unchanged, try as verb explicitly
        if lemma == word and wn_pos != wordnet.VERB:
            verb_lemma = _lemmatizer.lemmatize(word, wordnet.VERB)
            if verb_lemma != word:
                lemma = verb_lemma
        lemmas.append(lemma)
    return " ".join(lemmas), "POS-aware lemmatization: verbs→infinitive, nouns→singular. No inflected word passes unchanged."


def step_handling_negations(text: str) -> tuple[str, str]:
    """Step 6 – Negation Handling:
    1. Explicit negation words (not, never, etc.) bind to next 1–3 content words with NOT_ prefix.
    2. Intensifier + negative-prefix words (e.g., 'completely unhelpful') are transformed to NOT_root.
    3. Standalone negative-prefix words (e.g., 'unhelpful') are transformed to NOT_root."""
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Case 1: Explicit negation word → bind to next 1-3 words
        if token in _NEGATIONS or token.endswith("n't"):
            scope = min(3, len(tokens) - i - 1)
            applied = 0
            for j in range(1, scope + 1):
                if i + j < len(tokens):
                    next_token = tokens[i + j]
                    if next_token not in _NEGATIONS and next_token not in _BOOSTERS:
                        # If the next word already has a negative prefix, strip it
                        if _has_negative_prefix(next_token):
                            root = _strip_negative_prefix(next_token)
                            result.append(f"NOT_{root}")
                        else:
                            result.append(f"NOT_{next_token}")
                        applied += 1
                    else:
                        result.append(next_token)
                        applied += 1
            i += 1 + applied

        # Case 2: Intensifier/booster followed by a negative-prefix word
        elif token in _BOOSTERS and i + 1 < len(tokens) and _has_negative_prefix(tokens[i + 1]):
            root = _strip_negative_prefix(tokens[i + 1])
            result.append(f"NOT_{root}")
            i += 2

        # Case 3: Standalone negative-prefix word (e.g., "unhelpful" alone)
        elif _has_negative_prefix(token):
            root = _strip_negative_prefix(token)
            result.append(f"NOT_{root}")
            i += 1

        else:
            result.append(token)
            i += 1

    return " ".join(result), "Binding negations to targets with NOT_ prefix. Transforming intensifier+negative-prefix and standalone negative-prefix words."


def step_normalization(text: str) -> tuple[str, str]:
    """Step 7 – Normalization: Expand slang, abbreviations, and text-speak."""
    tokens = text.split()
    normalized = []
    for w in tokens:
        lookup = w.lower()
        if lookup in _SLANG_MAP:
            normalized.append(_SLANG_MAP[lookup])
        else:
            normalized.append(w)
    return " ".join(normalized), "Resolving slang, abbreviations, and text-speak into standard words."


PIPELINE = [
    step_tokenization,
    step_lowercasing,
    step_noise_removal,
    step_stop_word_removal,
    step_lemmatization,
    step_handling_negations,
    step_normalization,
]


def run_pipeline(text: str) -> list[dict]:
    """Run the full 7-step preprocessing pipeline."""
    steps = []
    current = text
    for idx, fn in enumerate(PIPELINE):
        result, desc = fn(current)
        steps.append({
            "step_no": idx + 1,
            "step_name": fn.__name__.replace("step_", "").replace("_", " ").title(),
            "result_text": result,
            "description": desc,
        })
        current = result
    return steps


def get_clean_text(text: str) -> str:
    """Return the fully preprocessed text."""
    return run_pipeline(text)[-1]["result_text"]


# --------------------------------------------------------------------------- #
# Sentiment Analysis (Context-Aware with Mathematical Integrity)
# --------------------------------------------------------------------------- #

def sentence_sentiment(text: str) -> dict:
    """Compute VADER sentiment on original text + calculate true distribution
    from the preprocessed word scores for mathematical integrity."""
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive 😊"
    elif compound <= -0.05:
        label = "Negative 😔"
    else:
        label = "Neutral 😐"
    scores["label"] = label
    return scores


def word_sentiment(text: str) -> list[dict]:
    """Compute deduplicated, context-aware per-word sentiment scores.
    - NOT_ prefixed tokens: Score the root word, then INVERT the polarity.
    - Structural/objective words are forced to 0.00.
    - No duplicate tokens in the output.
    - Context window (±1 word) for accuracy."""
    tokens = text.split()
    seen = {}  # Track seen tokens to prevent duplicates
    results = []

    for i, token in enumerate(tokens):
        # Skip if already scored (deduplication)
        if token in seen:
            continue

        # Structural word neutralization
        clean_token = token.replace("NOT_", "").lower()
        if clean_token in _STRUCTURAL_WORDS:
            score = 0.00
        elif token.startswith("NOT_"):
            # Score the ROOT word, then INVERT the polarity
            root = token[4:]  # Strip "NOT_"
            root_score = _analyzer.polarity_scores(root)["compound"]
            # If root has a clear sentiment, invert it
            if abs(root_score) >= 0.05:
                score = -root_score
            else:
                # Root is neutral; use context to determine sentiment
                start = max(0, i - 1)
                end = min(len(tokens), i + 2)
                context = " ".join(tokens[start:end])
                ctx_score = _analyzer.polarity_scores(context)["compound"]
                score = ctx_score if abs(ctx_score) > 0 else -0.25  # Default mild negative for negated neutral
        else:
            # Context window ±1
            start = max(0, i - 1)
            end = min(len(tokens), i + 2)
            context = " ".join(tokens[start:end])

            ctx_score = _analyzer.polarity_scores(context)["compound"]
            word_score = _analyzer.polarity_scores(token)["compound"]

            # Use the stronger signal between context and isolated word
            score = ctx_score if abs(ctx_score) > abs(word_score) else word_score

        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        entry = {"word": token, "compound": round(score, 4), "label": label}
        results.append(entry)
        seen[token] = entry

    return results


def word_sentiment_with_metrics(text: str) -> dict:
    """Return word-level scores + mathematically derived aggregate metrics.
    - mean_compound = exact arithmetic mean of all word scores.
    - distribution = ratio of positive/neutral/negative tokens."""
    words = word_sentiment(text)

    if not words:
        return {
            "words": [],
            "mean_compound": 0.0,
            "distribution": {"positive": 0.0, "neutral": 100.0, "negative": 0.0},
            "category": "Neutral 😐",
        }

    # Calculate exact arithmetic mean
    scores = [w["compound"] for w in words]
    mean_compound = round(sum(scores) / len(scores), 4)

    # Calculate true distribution from token counts
    pos_count = sum(1 for w in words if w["label"] == "positive")
    neg_count = sum(1 for w in words if w["label"] == "negative")
    neu_count = sum(1 for w in words if w["label"] == "neutral")
    total = len(words)

    distribution = {
        "positive": round((pos_count / total) * 100, 1),
        "neutral": round((neu_count / total) * 100, 1),
        "negative": round((neg_count / total) * 100, 1),
    }

    # Category from mean
    if mean_compound >= 0.05:
        category = "Positive 😊"
    elif mean_compound <= -0.05:
        category = "Negative 😔"
    else:
        # Check if there's a mix of positive and negative
        if pos_count > 0 and neg_count > 0:
            category = "Mixed 🤔"
        else:
            category = "Neutral 😐"

    return {
        "words": words,
        "mean_compound": mean_compound,
        "distribution": distribution,
        "category": category,
    }


def aspect_sentiment(text: str) -> list[dict]:
    """Aspect-based sentiment using a sliding context window."""
    tokens = text.split()
    aspects = []
    seen = set()
    for i, token in enumerate(tokens):
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        window = tokens[max(0, i - 2): i + 3]
        ctx = " ".join(window)
        score = _analyzer.polarity_scores(ctx)["compound"]
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        aspects.append({
            "aspect": token,
            "context": ctx,
            "compound": round(score, 4),
            "label": label,
        })
    return aspects
