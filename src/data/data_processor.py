import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import joblib
import os

class DataProcessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95, random_state=42)
        
    def load_data(self, file_path):
        return pd.read_csv(file_path)
        
    def clean_data(self, df):
        # Drop non-predictive identifier columns
        drop_cols = ['Student_ID', 'First_Name', 'Last_Name', 'Email', 'Grade']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Categorical NaN -> Mode
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        # Numerical NaN -> Median
        num_cols = df.select_dtypes(include='number').columns.tolist()
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        return df
        
    def engineer_features(self, df):
        if 'Final_Score' in df.columns and 'Midterm_Score' in df.columns:
            df['Score_Improvement'] = df['Final_Score'] - df['Midterm_Score']
        if 'Assignments_Avg' in df.columns and 'Quizzes_Avg' in df.columns:
            df['Assignment_Quiz_Avg'] = (df['Assignments_Avg'] + df['Quizzes_Avg']) / 2
        if 'Participation_Score' in df.columns and 'Projects_Score' in df.columns:
            df['Engagement_Score'] = (df['Participation_Score'] + df['Projects_Score']) / 2
        if 'Sleep_Hours_per_Night' in df.columns and 'Study_Hours_per_Week' in df.columns:
            df['Sleep_Study_Ratio'] = df['Sleep_Hours_per_Night'] / (df['Study_Hours_per_Week'] + 1)
        if 'Attendance (%)' in df.columns:
            df['High_Attendance'] = (df['Attendance (%)'] >= 75).astype(int)
        if 'Stress_Level (1-10)' in df.columns:
            df['Low_Stress'] = (df['Stress_Level (1-10)'] <= 4).astype(int)
            
        return df

    def encode_categorical(self, df):
        return pd.get_dummies(df, drop_first=False)
        
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca
        
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
        
    def save_preprocessors(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(save_dir, 'pca.pkl'))
        
    def load_preprocessors(self, load_dir):
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(load_dir, 'pca.pkl'))
