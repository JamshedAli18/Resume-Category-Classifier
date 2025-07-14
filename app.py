import streamlit as st
import joblib
import re

label_map = {
    0: 'ACCOUNTANT', 1: 'ADVOCATE', 2: 'AGRICULTURE', 3: 'APPAREL',
    4: 'ARTS', 5: 'AUTOMOBILE', 6: 'AVIATION', 7: 'BANKING',
    8: 'BPO', 9: 'BUSINESS-DEVELOPMENT', 10: 'CHEF', 11: 'CONSTRUCTION',
    12: 'CONSULTANT', 13: 'DESIGNER', 14: 'DIGITAL-MEDIA', 15: 'ENGINEERING',
    16: 'FINANCE', 17: 'FITNESS', 18: 'HEALTHCARE', 19: 'HR',
    20: 'INFORMATION-TECHNOLOGY', 21: 'PUBLIC-RELATIONS', 22: 'SALES', 23: 'TEACHER'
}

@st.cache_resource
def load_model():
    tfidf, model = joblib.load("resume_model.pkl")
    return tfidf, model

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

st.title("📄 Resume Category Classifier")
user_input = st.text_area("Paste your resume text here:", height=300)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please paste a resume first.")
    else:
        tfidf, model = load_model()
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        prediction_num = model.predict(vec)[0]
        prediction_label = label_map.get(int(prediction_num), "Unknown")
        st.success(f"Predicted Category: **{prediction_label}**")
