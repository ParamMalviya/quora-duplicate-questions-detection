# Quora Duplicate Questions Detection

Given two questions, predicts whether they're asking the same thing. Built on the
[Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs) dataset.

This is an early project of mine — classical ML, no deep learning. The interesting part
is the feature engineering: instead of letting a model learn representations, I hand-built
23 similarity features (including a tf-idf weighted cosine similarity) and fed them to a
RandomForest alongside bag-of-words.

**Live demo:** https://quora-duplicate-checker.streamlit.app/

## How it works

```
two raw questions
      |
      v  preprocess()      lowercase, expand contractions, strip HTML + punctuation
      |
      v  22 engineered features  +  tf-idf cosine similarity  +  bag-of-words (3000/question)
      |
      v  [22] + [1] + [3000 q1] + [3000 q2]  =  6023 numbers
      |
      v  RandomForestClassifier  ->  duplicate / not duplicate
```

### The 23 features

| Group | Features |
|---|---|
| Length & counts | `q1_len`, `q2_len`, `q1_num_words`, `q2_num_words` |
| Word overlap | `word_common`, `word_total`, `word_share` |
| Token ratios | `cwc_min/max`, `csc_min/max`, `ctc_min/max`, `last_word_eq`, `first_word_eq` |
| Length-derived | `abs_len_diff`, `mean_len`, `longest_substr_ratio` |
| Fuzzy string match | `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `token_set_ratio` |
| Similarity | `tfidf_cosine_sim` — cosine similarity weighted by how rare each shared word is |

Bag-of-words captures *which words appear*. The 23 features capture *how similar the two
strings are structurally* — which is the thing that actually matters for this problem.

## Results

Around **80% accuracy** on a held-out test set. Worth noting the classes aren't balanced —
roughly 63% of pairs are non-duplicates — so accuracy alone overstates things. `train.py`
prints a full classification report and confusion matrix for that reason.

## Running it

```bash
pip install -r requirements.txt
```

Download `train.csv` from the Kaggle competition link above and put it in the project root, then:

```bash
python train.py          # writes random_forest_model.pkl + count_vectorizer.pkl + idf_transformer.pkl
streamlit run app.py
```

## Notes on the code

- **`features.py` is shared by training and serving.** Both `train.py` and `app.py` import
  the same functions, so the two can't build features differently. An earlier version of
  this repo had them separate, and they drifted — the app was building the wrong number of
  features in the wrong order, so it couldn't actually run.
- **The vectorizer is fit after the train/test split**, not before. Fitting it on the full
  dataset first lets test-set vocabulary leak into training and inflates the score.
- **Tree depth is capped** (`max_depth=50`) mainly to keep the model pickle under GitHub's
  100MB file limit.
- Uses `rapidfuzz` rather than the older `fuzzywuzzy` — same scores, still maintained.

## Files

```
features.py       preprocessing + the 23 features, incl. tf-idf cosine sim (imported by both train and app)
train.py          trains the model, saves all three pickles (model, vectorizer, idf transformer)
app.py            streamlit UI
requirements.txt
```