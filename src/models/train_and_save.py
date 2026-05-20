import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score

def train_models():
    # Resolve paths relative to this script so the project is fully portable
    _base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path  = os.path.join(_base_dir, "data", "raw", "Students Performance Dataset.csv")

    print("Loading raw data...")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # 1. Target variable
    df['Pass_Fail'] = (df['Grade'] != 'F').astype(int)
    
    # 2. Drop non-predictive identifier columns
    drop_cols = ['Student_ID', 'First_Name', 'Last_Name', 'Email', 'Grade']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 3. Handle missing values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # 4. Feature Engineering
    df['Score_Improvement']   = df['Final_Score'] - df['Midterm_Score']
    df['Assignment_Quiz_Avg'] = (df['Assignments_Avg'] + df['Quizzes_Avg']) / 2
    df['Engagement_Score']    = (df['Participation_Score'] + df['Projects_Score']) / 2
    df['Sleep_Study_Ratio']   = df['Sleep_Hours_per_Night'] / (df['Study_Hours_per_Week'] + 1)
    df['High_Attendance']     = (df['Attendance (%)'] >= 75).astype(int)
    df['Low_Stress']          = (df['Stress_Level (1-10)'] <= 4).astype(int)
    
    # Save categorical modes and numerical medians for prediction pipeline imputation
    imputers = {
        'cat': {col: df[col].mode()[0] for col in cat_cols if col != 'Pass_Fail' and col in df.columns},
        'num': {col: df[col].median() for col in num_cols if col != 'Pass_Fail' and col in df.columns}
    }
    
    # 5. One-hot encoding
    # We will save the column list to align feature matrices at prediction time
    df_encoded = pd.get_dummies(df, drop_first=False)
    
    X = df_encoded.drop(columns=['Pass_Fail'])
    y = df_encoded['Pass_Fail']
    
    feature_names = X.columns.tolist()
    
    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    print(f"Random Forest - Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_val)
    xgb_acc = accuracy_score(y_val, xgb_preds)
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    print(f"XGBoost - Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
    
    # Save resources
    models_dir = os.path.join(_base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    joblib.dump(imputers, os.path.join(models_dir, "imputers.pkl"))
    
    # Save metadata
    metadata = {
        'rf_acc': rf_acc,
        'rf_auc': rf_auc,
        'xgb_acc': xgb_acc,
        'xgb_auc': xgb_auc
    }
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.pkl"))
    print("All models and training metadata saved successfully!")

if __name__ == "__main__":
    train_models()
