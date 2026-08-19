import pickle
import streamlit as st

from features import make_input_vector

st.set_page_config(page_title="Quora Duplicate Question Checker", page_icon="❓")

# cache_resource so streamlit loads the pickles once instead of on every click
@st.cache_resource
def load_artifacts():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("count_vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return model, cv

model, cv, idf_transformer = load_artifacts()
...
features = make_input_vector(q1, q2, cv, idf_transformer)

st.title("Quora Duplicate Question Checker")
st.write("Checks whether two questions are asking the same thing.")

q1 = st.text_input("Question 1", placeholder="How do I learn Python?")
q2 = st.text_input("Question 2", placeholder="What is the best way to learn Python?")

if st.button("Check if Duplicate"):
    if not q1.strip() or not q2.strip():
        st.warning("Enter both questions first.")
    else:
        # one function call does preprocess -> 22 features -> bow -> stack.
        # importing it from features.py is the whole point: training uses the
        # exact same code, so the app can't build the vector differently anymore
        features = make_input_vector(q1, q2, cv)

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.success(f"Duplicate  (confidence {proba:.0%})")
        else:
            st.error(f"Not Duplicate  (confidence {1 - proba:.0%})")

        st.caption("Trained on the Quora Question Pairs dataset. ~80% accuracy, so it does get things wrong.")