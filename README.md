# Spam_email
```markdown
# Email Spam Classifier using TF-IDF and LinearSVC

This repository contains a high-performance, production-ready machine learning pipeline for classifying emails as **Ham (legitimate)** or **Spam**. The model achieves an outstanding **99.65% accuracy** on holdout test data.

## 📋 Project Overview

Identifying spam email is a classic Natural Language Processing (NLP) problem. This project walks through the complete process of:
- **Exploratory Data Analysis (EDA):** Class distribution and description of data.
- **Feature Extraction:** Leveraging Term Frequency-Inverse Document Frequency (`TF-IDF`) vectorization.
- **Model Selection & Tuning:** Exploring `SVC`, `LinearSVC`, and performing hyperparameter tuning using `GridSearchCV`.
- **Validation:** Applying Stratified `K-Fold` Cross-Validation to guarantee generalization.
- **Deployment Readiness:** Serializing the finalized pipeline using `joblib` for easy inference.

## 📊 Dataset

We utilize the **Spam Email Dataset** from Kaggle, containing **5,728 unique emails**:
- **Ham (0):** 4,360 samples
- **Spam (1):** 1,368 samples

## 🚀 Key Experiments & Performance

We compared multiple feature engineering and model combinations on the holdout test set (20% split):

| Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline SVC (RBF Kernel)** | 99.39% | 98.55% | 98.91% | 98.72% |
| **LinearSVC + Word Unigrams** | 99.56% | 98.91% | 99.27% | 99.09% |
| **LinearSVC + Char n-grams (3-5)** | 99.13% | 97.48% | 98.91% | 98.19% |
| **LinearSVC + Word Bigrams** | 99.39% | 98.55% | 98.91% | 98.72% |
| **Tuned LinearSVC (C=100) 🏆** | **99.65%** | **98.91%** | **99.64%** | **99.27%** |

### 5-Fold Cross Validation Results (F1-Score)
- **Baseline SVC:** `0.9864` (± 0.0045)
- **LinearSVC (C=1):** `0.9893` (± 0.0043)
- **LinearSVC (C=100):** `0.9901` (± 0.0036)

## 🛠️ Installation & Quickstart

To run this project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
pip install -r requirements.txt
```

### Requirements
```text
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

## 💻 Inference with Saved Model

The final model is saved as `sms_spam_classifier.pkl`. You can load and run predictions on new, unseen emails with just a few lines of code:

```python
import joblib

# Load the saved model pipeline
model = joblib.load("sms_spam_classifier.pkl")

# Input messages
messages = [
    "Congratulations! You have won a free prize. Click here to claim.",
    "Hey, are we still meeting up for dinner tonight?"
]

# Predict (0 = Ham, 1 = Spam)
predictions = model.predict(messages)

for msg, pred in zip(messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}]: {msg}")
```

## 📈 Key Technical Decisions
- **Optimized Solver Convergence:** Solved the `ConvergenceWarning` when training `LinearSVC` with a high regularization value ($C=100$) by increasing `max_iter` to `10000`.
- **Unified Pipeline:** Both the TF-IDF feature extractor and classifier are packaged inside a single scikit-learn pipeline, ensuring text preprocessing and modeling step-by-step consistency during deployment.
