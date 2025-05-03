import streamlit as st
import pickle
import numpy as np

# Load trained RandomForest model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the same CountVectorizer used in training
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

st.title("Quora Duplicate Question Checker")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check if Duplicate"):
    q1_len = len(q1)
    q2_len = len(q2)
    q1_num_words = len(q1.split())
    q2_num_words = len(q2.split())

    # Transform questions using CountVectorizer
    q1_vec = cv.transform([q1]).toarray()
    q2_vec = cv.transform([q2]).toarray()

    # Combine all features
    features = np.hstack((q1_vec, q2_vec, np.array([[q1_len, q2_len, q1_num_words, q2_num_words]])))

    # Make prediction
    prediction = model.predict(features)[0]
    result = "✅ Duplicate" if prediction == 1 else "❌ Not Duplicate"

    st.success(f"Result: {result}")
