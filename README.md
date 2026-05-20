# 🎓 Scholara — Academic Intelligence Platform

> A smart, local AI-powered academic advisor built with Streamlit, NLP, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What is Scholara?

**Scholara** is an intelligent academic advising platform built for university students. It combines local NLP (sentence embeddings), ML-based grade prediction (XGBoost + Random Forest), and a premium Streamlit UI to deliver a personalized academic experience — **fully offline, no API keys required**.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🤖 **AI Academic Advisor** | Multi-turn Arabic/English chatbot that understands grades, asks follow-ups, and gives personalized advice |
| 📈 **Grade Predictor** | Dual ML models (XGBoost + Random Forest) predict Pass/Fail probability from your inputs |
| 🧮 **Score Calculator** | Instantly calculates how much you need in the final exam to reach your target |
| 📊 **Advanced Analytics** | Dataset-level visualizations: distributions, correlations, model evaluations |
| 💡 **Study Tips** | Personalized, score-based study recommendations in Arabic and English |
| 🌍 **Bilingual** | Full Arabic (Egyptian colloquial) and English support with language toggle |

---

## 🏗️ Architecture

```
Scholara/
├── app.py                          # Main Streamlit application
├── src/
│   ├── nlp/
│   │   └── nlp_processor.py        # Multilingual sentence-embedding NLP engine
│   ├── conversation/
│   │   └── conversation_manager.py # Multi-turn dialogue orchestration
│   ├── recommendation/
│   │   └── recommendation_engine.py # Score-based advice & prediction responses
│   ├── data/
│   │   └── data_processor.py       # Feature engineering & data cleaning pipeline
│   └── models/
│       ├── model_trainer.py        # Model training abstraction
│       └── train_and_save.py       # Training script (run once to generate models)
├── models/                         # Trained ML model artifacts (generated)
├── data/
│   └── raw/                        # Place your dataset CSV here
├── notebooks/                      # Exploratory data analysis notebooks
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/BassemRamdan/Scholara.git
cd Scholara
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place the dataset file in the `data/raw/` directory:
```
data/raw/Students Performance Dataset.csv
```

### 4. Train the ML Models

Run this **once** to train XGBoost and Random Forest models and save them:
```bash
python src/models/train_and_save.py
```

This will generate the following in `models/`:
- `rf_model.pkl` — Random Forest classifier
- `xgb_model.pkl` — XGBoost classifier
- `feature_names.pkl` — Feature alignment list
- `model_metadata.pkl` — Accuracy & AUC metrics

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 🤖 NLP Pipeline

Scholara uses **`paraphrase-multilingual-MiniLM-L12-v2`** (Sentence Transformers) for semantic intent detection. The model runs fully locally — no internet connection or API keys needed after the first model download.

- Handles **Egyptian Arabic colloquial**, **Arabizi**, **English**, typos, and mixed-language input
- Intent detection via **cosine similarity** against prototype embeddings
- Score extraction supports both **Arabic-Indic (٠١٢٣)** and Western digits

---

## 🧠 ML Models

Both classifiers are trained to predict **Pass/Fail** from academic and behavioral features:

| Model | Accuracy | AUC |
|---|---|---|
| XGBoost | ~97% | ~99% |
| Random Forest | ~97% | ~99% |

**Features used:** Midterm Score, Final Score, Assignments, Quizzes, Projects, Participation, Attendance, Study Hours, Sleep, Stress Level, Department, GPA components, and engineered features.

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
sentence-transformers
matplotlib
seaborn
arabic-reshaper
python-bidi
scipy
```

---

## 👨‍💻 Author

**Bassem Ramdan** — Intelligent Programming Project, ANU

---

## 📄 License

This project is licensed under the MIT License.
