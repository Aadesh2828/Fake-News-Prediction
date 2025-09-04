import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Fake News Detection App")

# Input text box
user_input = st.text_area("Enter News Article Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        # Transform text
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]
        
        # Show result
        if prediction == 1:  # assuming 1 = True News, 0 = Fake
            st.success("✅ This looks like a --True News-- article.")
        else:
            st.error("❌ This looks like a --Fake News-- article.")
