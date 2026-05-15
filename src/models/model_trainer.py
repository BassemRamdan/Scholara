import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=400,
                max_depth=20,
                min_samples_leaf=2,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=0.1,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        }
        self.best_model = None
        self.best_model_name = None
        
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        results = []
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            acc = accuracy_score(y_val, preds)
            auc = roc_auc_score(y_val, proba) if proba is not None else 0
            
            results.append({'Model': name, 'Val Accuracy': acc, 'ROC-AUC': auc})
            
        results_df = pd.DataFrame(results).set_index('Model').sort_values('Val Accuracy', ascending=False)
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        return results_df
        
    def save_best_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if self.best_model is not None:
            joblib.dump(self.best_model, os.path.join(save_dir, f'best_model_{self.best_model_name.replace(" ", "_").lower()}.pkl'))
            
    def load_model(self, model_path):
        self.best_model = joblib.load(model_path)
