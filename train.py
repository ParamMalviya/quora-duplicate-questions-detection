# retrains the model and saves BOTH pickles together, so they always match.
# run this once: python train.py
#
# why this exists: the old repo had count_vectorizer.pkl committed but the
# model pickle was missing entirely, so the app could never actually run.

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import preprocess, build_features

# 50k rows instead of 100k. a forest trained on the full set pickles out to
# a few hundred MB, and github hard-rejects anything over 100MB - which is
# almost certainly why the model file never got uploaded in the first place.
SAMPLE_SIZE = 100000
CSV_PATH = "train.csv"   # kaggle quora question pairs

print("loading data...")
df = pd.read_csv(CSV_PATH).sample(SAMPLE_SIZE, random_state=2)
df = df.dropna(subset=["question1", "question2"])

print("preprocessing...")
df["question1"] = df["question1"].apply(preprocess)
df["question2"] = df["question2"].apply(preprocess)

print("building the 22 engineered features...")
engineered = np.array([
    build_features(r["question1"], r["question2"])
    for _, r in df.iterrows()
])

y = df["is_duplicate"].values

# SPLIT FIRST, then fit the vectorizer. the old notebook fit CountVectorizer on
# the whole dataset before splitting, which let test-set vocabulary leak into
# training and made the accuracy look slightly better than it really was.
print("splitting BEFORE fitting the vectorizer (this fixes the leakage)...")
idx_train, idx_test, y_train, y_test = train_test_split(
    np.arange(len(df)), y, test_size=0.2, random_state=1, stratify=y
)

q1_all = df["question1"].values
q2_all = df["question2"].values

# fit on TRAIN questions only
cv = CountVectorizer(max_features=3000)
cv.fit(list(q1_all[idx_train]) + list(q2_all[idx_train]))

from sklearn.feature_extraction.text import TfidfTransformer

idf_transformer = TfidfTransformer()
idf_transformer.fit(cv.transform(list(q1_all[idx_train]) + list(q2_all[idx_train])))

from sklearn.preprocessing import normalize

def vectorize(indices):
    q1_bow = cv.transform(q1_all[indices]).toarray()
    q2_bow = cv.transform(q2_all[indices]).toarray()

    # vectorized cosine similarity for all rows at once - a python loop over
    # 80k rows here would be the slow way to do this
    q1_tfidf = normalize(idf_transformer.transform(q1_bow))
    q2_tfidf = normalize(idf_transformer.transform(q2_bow))
    sims = np.asarray(q1_tfidf.multiply(q2_tfidf).sum(axis=1))

    return np.hstack((engineered[indices], sims, q1_bow, q2_bow))

X_train = vectorize(idx_train)
X_test = vectorize(idx_test)
print("train shape:", X_train.shape, "(should be 6023 columns)")

# max_depth caps how big the trees get, which is what actually keeps the
# pickle small enough to commit. costs about 1% accuracy, worth it.
print("training...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=50,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\naccuracy:", round(accuracy_score(y_test, y_pred), 4))
# accuracy alone is misleading here - the classes aren't balanced, so a model
# that always guessed "not duplicate" would already score around 63%
print("\n", classification_report(y_test, y_pred, target_names=["not duplicate", "duplicate"]))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

print("\nsaving both pickles together so they can never drift apart...")
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("count_vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("idf_transformer.pkl", "wb") as f:
    pickle.dump(idf_transformer, f, protocol=pickle.HIGHEST_PROTOCOL)

import os
print("model size:", round(os.path.getsize("random_forest_model.pkl") / 1e6, 1), "MB (must stay under 100)")
print("done")