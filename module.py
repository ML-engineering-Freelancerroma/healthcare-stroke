import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)


df = pd.read_csv('health_train.csv')
X = df.drop(columns=['stroke'])
y = df['stroke']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_df = pd.DataFrame(
    {
        'Model': [],
        'Accuracy': [],
        'Recall': [],
        'ROC-AUC': [],
        'PR-AUC': [],
    }
).astype(
    {
        'Model': str,
        'Accuracy': float,
        'Recall': float,
        'ROC-AUC': float,
        'PR-AUC': float,
    }
)


def evaluate_and_append(model_name, best_estimator, X, y, cv, results_df):

    y_pred = cross_val_predict(
        best_estimator, X, y, cv=cv, method='predict', n_jobs=-1
    )
    y_proba = cross_val_predict(
        best_estimator, X, y, cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_proba),
        "PR-AUC": average_precision_score(y, y_proba)
    }

    new_row = pd.DataFrame([metrics])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df

# For SMOTENC, CatBoost


cat_features = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'Residence_type',
    'age_old',
    'age_child',
    'bmi_fat'
]
