"""
Breast Cancer Detection Model
CS210 Project

Dataset: Breast Cancer Wisconsin (Diagnostic) - wdbc.data
Goal: Classify tumors as Malignant (M) or Benign (B) using machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# loading the dataset

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_error', 'texture_error', 'perimeter_error', 'area_error',
    'smoothness_error', 'compactness_error', 'concavity_error',
    'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
    'radius_extreme', 'texture_extreme', 'perimeter_extreme', 'area_extreme',
    'smoothness_extreme', 'compactness_extreme', 'concavity_extreme',
    'concave_points_extreme', 'symmetry_extreme', 'fractal_dimension_extreme'
] # this is needed b/c the dataset file has no header row, so must be assigned manually

columns = ['id', 'diagnosis'] + feature_names
df = pd.read_csv('wdbc.data', header=None, names=columns)
df.drop('id', axis=1, inplace=True)

# get an overivew and general insights on dataset
print("Overview of Dataset")
print(f"Shape: {df.shape}")
print(f"\nClass distribution:\n{df['diagnosis'].value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}") #dataset should be clean, but making sure there are no leftover empty vals
print("\nHead:")
print(df.head())

# data analysis and visualization before preprocessing

# encode labels: M=1 for malignant, B=0 for benign
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

malignant = df[df['diagnosis'] == 1] #df filtered to only malignant entries
benign = df[df['diagnosis'] == 0] # df filtered to only benign entries


# preprocessing the data for modeling
X = df.drop('diagnosis', axis=1) # drop diagnosis column for feature df
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #split data into train and test sets (stratify =y makes sure the class distribution is preserved in both sets to avoid skew in results)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# train and evaluate multiple models (with hyperparameter tuning for main ones)

# --- 4a. Hyperparameter tuning via GridSearchCV (on training set) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Hyperparameter Tuning (GridSearchCV, 5-fold CV) ===")

lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=10000, random_state=42),
                       lr_params, cv=cv, scoring='recall', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)
print(f"  Logistic Regression best C     : {lr_grid.best_params_['C']}")

rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=cv, scoring='recall', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
print(f"  Random Forest best params      : {rf_grid.best_params_}")

svm_params = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
                        svm_params, cv=cv, scoring='recall', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
print(f"  SVM best params                : {svm_grid.best_params_}")

gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
             'max_depth': [3, 5]}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                       gb_params, cv=cv, scoring='recall', n_jobs=-1)
gb_grid.fit(X_train_scaled, y_train)
print(f"  Gradient Boosting best params  : {gb_grid.best_params_}")

# --- 4b. All models (tuned + additional) ---
models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'Random Forest':       rf_grid.best_estimator_,
    'SVM':                 svm_grid.best_estimator_,
    'Gradient Boosting':   gb_grid.best_estimator_,
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
}

# Fit models not yet fit (KNN, DT)
for name in ['KNN', 'Decision Tree']:
    models[name].fit(X_train_scaled, y_train)

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # 5-fold cross-validation on full scaled data
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=cv, scoring='recall')
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'fpr':       fpr,
        'tpr':       tpr,
        'auc':       auc(fpr, tpr),
        'cv_recall_mean': cv_scores.mean(),
        'cv_recall_std':  cv_scores.std(),
    }

# evaluate all models

print("\n Model Performance (Test Set + 5-Fold CV Recall)")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8} {'CV Recall':>12}")
print("-" * 84)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>10.3f} {r['precision']:>10.3f} "
          f"{r['recall']:>10.3f} {r['f1']:>8.3f} {r['auc']:>8.3f} "
          f"  {r['cv_recall_mean']:.3f}±{r['cv_recall_std']:.3f}")

print()
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r['y_pred'],
                                 target_names=['Benign', 'Malignant']))

# visualize model results
# confusion matrices (2 rows × 3 cols for 6 models)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, (name, r) in enumerate(results.items()):
    ax = axes[i]
    cm = confusion_matrix(y_test, r['y_pred'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name}\nAcc={r["accuracy"]:.3f}  Recall={r["recall"]:.3f}')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]),
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[row, col] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("Saved: confusion_matrices.png")

# ROC curves — all models
fig, ax = plt.subplots(figsize=(9, 7))
plot_colors = ['steelblue', 'darkorange', 'green', 'purple', 'brown', 'crimson']
for (name, r), color in zip(results.items(), plot_colors):
    ax.plot(r['fpr'], r['tpr'], color=color, lw=2,
            label=f'{name} (AUC = {r["auc"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models'); ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
print("Saved: roc_curves.png")

# Cross-validation recall comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
cv_means = [results[n]['cv_recall_mean'] for n in names]
cv_stds  = [results[n]['cv_recall_std']  for n in names]
bars = ax.bar(range(len(names)), cv_means, yerr=cv_stds, capsize=5,
              color='steelblue', alpha=0.8)
ax.set_ylim([0.85, 1.01])
ax.set_ylabel('CV Recall (mean ± std)')
ax.set_title('5-Fold Cross-Validation Recall — All Models')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha='right')
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_recall_comparison.png', dpi=150)
plt.show()

# random forest feature importance (only doing for RF since it's the most interpretable model and has built-in feature importance)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue', ax=ax)
ax.set_title('Top 20 Feature Importances (Random Forest)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Saved: feature_importance.png")

#  summary statistics and insights
# Best by CV recall (more reliable than single split)
best_model_name = max(results, key=lambda n: results[n]['cv_recall_mean'])
print(f"\n=== Best Model by CV Recall ===")
print(f"  {best_model_name}")
print(f"  Accuracy : {results[best_model_name]['accuracy']:.3f}")
print(f"  Precision: {results[best_model_name]['precision']:.3f}")
print(f"  Recall   : {results[best_model_name]['recall']:.3f}")
print(f"  F1       : {results[best_model_name]['f1']:.3f}")
print(f"  AUC      : {results[best_model_name]['auc']:.3f}")
print(f"  CV Recall: {results[best_model_name]['cv_recall_mean']:.3f} "
      f"± {results[best_model_name]['cv_recall_std']:.3f}")

print("\nTop 5 most important features (Random Forest):")
for feat, score in importances.sort_values(ascending=False).head(5).items():
    print(f"  {feat:<30} {score:.4f}")
