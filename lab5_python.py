from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

csv = Path(kagglehub.dataset_download('mlg-ulb/creditcardfraud')) / 'creditcard.csv'
df = pd.read_csv(csv)
X = df.drop(columns='Class')
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print('Размер данных:', df.shape)
print('Мошеннических транзакций:', y.sum())
print('Тест:', len(y_test), '| мошеннических:', y_test.sum())

class CompatibleCatBoost(CatBoostClassifier, ClassifierMixin, BaseEstimator):
    pass

spaces = {
    'Bagging': (
        BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_jobs=1, random_state=42),
        {
            'n_estimators': [20, 40, 60],
            'estimator__max_depth': [6, 10, 15],
            'estimator__min_samples_leaf': [1, 2, 5],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.6, 0.8, 1.0]
        }
    ),
    'Random Forest': (
        RandomForestClassifier(n_jobs=1, random_state=42),
        {
            'n_estimators': [50, 100, 150],
            'max_depth': [8, 12, None],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 0.5, None],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
    ),
    'LightGBM': (
        LGBMClassifier(n_jobs=1, verbosity=-1, random_state=42),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.03, 0.05, 0.1],
            'num_leaves': [7, 15, 31],
            'min_child_samples': [10, 30, 60],
            'scale_pos_weight': [1, 10, 100]
        }
    ),
    'CatBoost': (
        CompatibleCatBoost(thread_count=1, verbose=False, random_seed=42),
        {
            'iterations': [150, 300, 500],
            'depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 10],
            'auto_class_weights': ['None', 'SqrtBalanced', 'Balanced']
        }
    )
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
thresholds = np.r_[0.001, 0.005, np.arange(0.01, 1.0, 0.01)]
rows = []
fitted = {}

for name, (model, params) in spaces.items():
    search = GridSearchCV(
        model, params, scoring='f1_macro', cv=cv,
        n_jobs=2, pre_dispatch=2, refit=True, verbose=1, error_score='raise'
    )
    search.fit(X_train, y_train)
    p_val = search.best_estimator_.predict_proba(X_val)[:, 1]
    threshold = max(thresholds, key=lambda t: f1_score(y_val, p_val >= t, average='macro'))
    val_f1 = f1_score(y_val, p_val >= threshold, average='macro')
    rows.append([name, search.best_score_, threshold, val_f1])
    fitted[name] = (search.best_estimator_, threshold)
    print(name, '| лучшие параметры:', search.best_params_)

results = pd.DataFrame(rows, columns=['Модель', 'CV Macro F1', 'Порог', 'Val Macro F1'])
results = results.sort_values('Val Macro F1', ascending=False)
display(results.round(4))

best = results.iloc[0]['Модель']
model, threshold = fitted[best]
y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

print('Лучшая модель:', best)
print('Порог:', round(float(threshold), 4))
print('Test Macro F1:', round(f1_score(y_test, y_pred, average='macro'), 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Легальная', 'Мошенничество'],
    cmap='Blues', values_format='d'
)
plt.title(f'Матрица ошибок — {best}')
plt.show()
