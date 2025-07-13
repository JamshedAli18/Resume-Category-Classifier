# 📄 Resume Category Classifier

A simple end‑to‑end project that trains a TF‑IDF + Random Forest model to predict professional resume categories (e.g. **SALES**, **FINANCE**, **ENGINEERING**, etc.), and serves predictions with a **Streamlit** web app.

---

##  Features

| Feature | Description |
|--------|-------------|
| **Colab Training Notebook** | `train.ipynb` trains TF‑IDF + Random Forest, evaluates, and saves `resume_model.pkl`. |
| **Ready‑to‑use model** | You can train and save the model in Colab, then download and use in Streamlit. |
| **Streamlit app** | `app.py` loads the model and offers a paste‑in‑resume interface for prediction. |
| **Label mapping** | `label_map.py` converts numeric predictions → human‑readable labels (e.g. `22 → SALES`). |
| **Testing Examples** | Includes resume samples across categories to test the classifier. |

---

## 🗂️ Directory Structure

```text
.
├── app.py                # Streamlit interface
├── resume_model.pkl      # Saved TF‑IDF vectorizer + RF model
├── requirements.txt      # Python deps
