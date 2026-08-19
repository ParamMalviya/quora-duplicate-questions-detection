# all feature engineering lives here, in ONE place.
# train.py and app.py both import from this file, so they can never drift apart.
# that drift was the actual bug last time: the app built features differently than training did.

import re
import difflib

import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.preprocessing import normalize

# streamlit cloud starts with a bare nltk, so grab stopwords if they aren't there yet
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# build the stopword SET once at import time.
# the old notebook rebuilt this list inside the per-row function, so it ran 100k times.
# a set also makes lookups instant instead of scanning a list every single time.
STOP_WORDS = set(stopwords.words("english"))

SAFE_DIV = 0.0001

# the exact order training used. app.py must produce these in the SAME order,
# otherwise the model gets the right numbers in the wrong slots.
#
# heads up to myself: these 23 names come from TWO different functions, not one.
#   indices 0-21 (up to "token_set_ratio")  <- build_features()
#   index 22 ("tfidf_cosine_sim")           <- tfidf_cosine_similarity(),
#                                              only stitched together in make_input_vector()
# so build_features() alone returns 22 values, not 23. worth remembering if I ever
# use FEATURE_NAMES to label model.feature_importances_ - everything past index 21
# would silently shift by one.
FEATURE_NAMES = [
    "q1_len", "q2_len", "q1_num_words", "q2_num_words",
    "word_common", "word_total", "word_share",
    "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max",
    "last_word_eq", "first_word_eq",
    "abs_len_diff", "mean_len", "longest_substr_ratio",
    "fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio",
    "tfidf_cosine_sim",
]

CONTRACTIONS = { 
"ain't": "am not",
"aren't": "are not",
"can't": "can not",
"can't've": "can not have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def preprocess(q):
    """clean up a question the same way training did. this is the step app.py
    was skipping entirely, which is why serving text never matched training text."""
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # expand contractions so "don't" and "do not" look like the same thing
    q = ' '.join(CONTRACTIONS.get(word, word) for word in q.split())
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # strip html tags. I name html.parser explicitly (stdlib, so no extra dependency
    # to install on streamlit cloud) which also stops bs4 warning about an unspecified parser
    q = BeautifulSoup(q, "html.parser").get_text()

    # kill punctuation
    q = re.sub(re.compile(r'\W'), ' ', q).strip()
    return q


def build_features(q1, q2):
    """take two RAW questions, return the 22 features in FEATURE_NAMES order.
    call preprocess first - this function assumes it already ran."""
    feats = []

    # --- basic length/count features (4) ---
    feats.append(len(q1))
    feats.append(len(q2))
    feats.append(len(q1.split()))
    feats.append(len(q2.split()))

    # --- word overlap features (3) ---
    w1 = set(word.lower().strip() for word in q1.split(" "))
    w2 = set(word.lower().strip() for word in q2.split(" "))
    word_common = len(w1 & w2)
    word_total = len(w1) + len(w2)
    word_share = round(word_common / word_total, 2) if word_total else 0.0
    feats.append(word_common)
    feats.append(word_total)
    feats.append(word_share)

    # --- token features (8) ---
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        # empty question, everything below would divide by zero
        feats.extend([0.0] * 8)
    else:
        q1_words = set(w for w in q1_tokens if w not in STOP_WORDS)
        q2_words = set(w for w in q2_tokens if w not in STOP_WORDS)
        q1_stops = set(w for w in q1_tokens if w in STOP_WORDS)
        q2_stops = set(w for w in q2_tokens if w in STOP_WORDS)

        common_word_count = len(q1_words & q2_words)
        common_stop_count = len(q1_stops & q2_stops)
        common_token_count = len(set(q1_tokens) & set(q2_tokens))

        feats.append(common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV))
        feats.append(common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV))
        feats.append(common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV))
        feats.append(common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV))
        feats.append(common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV))
        feats.append(common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV))
        feats.append(int(q1_tokens[-1] == q2_tokens[-1]))
        feats.append(int(q1_tokens[0] == q2_tokens[0]))

    # --- length features (3) ---
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        feats.extend([0.0] * 3)
    else:
        feats.append(abs(len(q1_tokens) - len(q2_tokens)))
        feats.append((len(q1_tokens) + len(q2_tokens)) / 2)
        matcher = difflib.SequenceMatcher(None, q1, q2)
        match = matcher.find_longest_match(0, len(q1), 0, len(q2))
        feats.append(match.size / (min(len(q1), len(q2)) + 1))

    # --- fuzzy features (4) ---
    # rapidfuzz instead of fuzzywuzzy: same numbers, actively maintained, way faster
    feats.append(fuzz.QRatio(q1, q2))
    feats.append(fuzz.partial_ratio(q1, q2))
    feats.append(fuzz.token_sort_ratio(q1, q2))
    feats.append(fuzz.token_set_ratio(q1, q2))

    return feats

def tfidf_cosine_similarity(q1, q2, cv, idf_transformer):
    """cosine similarity between q1 and q2, weighted by how RARE each shared
    word is - not just whether it's shared. sharing "python" should count for
    more than sharing "what". none of the other 22 features capture this."""
    bow = cv.transform([q1, q2])
    tfidf = normalize(idf_transformer.transform(bow))
    return float(tfidf[0].multiply(tfidf[1]).sum())

def make_input_vector(q1_raw, q2_raw, cv, idf_transformer):
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)

    engineered = np.array(build_features(q1, q2)).reshape(1, -1)
    sim = np.array([[tfidf_cosine_similarity(q1, q2, cv, idf_transformer)]])
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((engineered, sim, q1_bow, q2_bow))