import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# Educational purpose
DATA_PATH = 'health_train.csv'

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['stroke'])
y = df['stroke']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# For SMOTENC, CatBoost
CAT_FEATURES = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'Residence_type',
    'age_old',
    'age_child',
    'bmi_fat'
]


def evaluate_and_append(model_name, best_estimator, X, y, cv, results_df):
    """
    Evaluate the model using cross-validation
    and append results to the DataFrame
    """

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


class StrokeDataTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for new data
    """

    def __init__(self):
        self.bmi_median = None
        self.gender_encoder = LabelEncoder()
        self.work_type_cols_after_dummies = None
        self.smoke_cols_after_dummies = None

    def fit(self, df, y=None):
        df_fit = df.copy()
        self.bmi_median = df_fit['bmi'].median()
        df_fit['gender'] = df_fit['gender'].str.capitalize()
        df_fit = df_fit[df_fit['gender'] != 'Other']
        self.gender_encoder.fit(df_fit['gender'])

        dummies_work = pd.get_dummies(
            df_fit, columns=['work_type'],
            prefix='work',
            drop_first=True,
            dtype=int
        )

        self.work_type_cols_after_dummies = [
            col for col in dummies_work.columns if col.startswith('work_')
        ]

        dummies_smoke = pd.get_dummies(
            df_fit, columns=['smoking_status'],
            prefix='smoke',
            drop_first=False,
            dtype=int
        )

        smoke_cols = [
            col for col in dummies_smoke.columns if col.startswith('smoke_')
        ]
        self.smoke_cols_after_dummies = [
            col for col in smoke_cols if col != 'smoke_never smoked'
        ]

        return self

    def transform(self, df):
        df_t = df.copy()
        if 'id' in df_t.columns:
            df_t = df_t.drop('id', axis=1)
        df_t['bmi'] = df_t['bmi'].fillna(self.bmi_median)
        df_t['gender'] = df_t['gender'].str.capitalize()
        df_t = df_t[df_t['gender'] != 'Other']
        df_t['gender'] = self.gender_encoder.transform(df_t['gender'])
        df_t['ever_married'] = (df_t['ever_married'] == 'Yes').astype(int)

        df_t = pd.get_dummies(
            df_t, columns=['work_type'],
            prefix='work',
            drop_first=True,
            dtype=int
        )

        df_t['Residence_type'] = (
            df_t['Residence_type'] == 'Urban'
        ).astype(int)

        df_t = pd.get_dummies(
            df_t, columns=['smoking_status'],
            prefix='smoke',
            drop_first=False,
            dtype=int
        )

        df_t['age_old'] = (df_t['age'] >= 50).astype(int)
        df_t['age_child'] = (df_t['age'] < 18).astype(int)
        df_t['bmi_fat'] = (df_t['bmi'] >= 30).astype(int)
        df_t['age_hypertension'] = df_t['age'] * df_t['hypertension']

        binary_cols = [
            'gender', 'hypertension', 'heart_disease', 'ever_married',
            'Residence_type', 'age_old', 'age_child', 'bmi_fat'
        ]

        work_cols = [col for col in df_t.columns if col.startswith('work_')]
        smoke_cols = [col for col in df_t.columns if col.startswith('smoke_')]

        int8_cols = binary_cols + work_cols + smoke_cols
        int8_cols = [c for c in int8_cols if c in df_t.columns]
        df_t[int8_cols] = df_t[int8_cols].astype('int8')

        float_cols = ['age', 'avg_glucose_level', 'bmi', 'age_hypertension']
        float_cols = [c for c in float_cols if c in df_t.columns]
        df_t[float_cols] = df_t[float_cols].astype('float32')

        return df_t
