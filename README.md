# Student Performance Prediction

This project aims to analyze and predict student performance based on the Students Performance Dataset, which contains records for approximately 5,000 students.

## Project Parts
1. **Part 1:** Data Cleaning & Exploratory Data Analysis (EDA).
2. **Part 2:** Model Training & Evaluation (Logistic Regression, Random Forest, XGBoost).
3. **Part 3:** Natural Language Processing (NLP) — Entity Extraction from Egyptian Arabic.

## Folder Structure
The project is organized following best practices for Machine Learning and Data Science projects:

```text
EduGuide/
│
├── data/                   # Data directory
│   ├── raw/                # Original raw data (Students Performance Dataset.csv)
│   └── processed/          # Cleaned and processed data ready for training
│
├── notebooks/              # Jupyter Notebooks for exploration and analysis
│   └── student_performance_prediction.ipynb
│
├── src/                    # Source code scripts for the project
│   ├── data/               # Scripts for data cleaning and preprocessing
│   ├── models/             # Scripts for model training and evaluation
│   ├── nlp/                # Natural Language Processing (NLP) scripts
│   └── recommendation/     # Recommendation system scripts
│
├── app/                    # Streamlit UI application files
│
├── models/                 # Saved trained models (.pkl, .joblib, etc.)
│
├── reports/                # Generated reports and visualizations
│
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

## How to Use
- **Data:** Place any new raw data in the `data/raw/` directory.
- **Exploration:** Conduct EDA and experimental modeling inside the `notebooks/` directory.
- **Production:** Write clean, modular, and reusable code in the `src/` directory for automated execution.
