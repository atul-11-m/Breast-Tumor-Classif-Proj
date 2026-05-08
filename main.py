"""
Breast Cancer Detection Model
CS210 Project

Dataset: Breast Cancer Wisconsin (Diagnostic) - wdbc.data
Goal: Classify tumors as Malignant (M) or Benign (B) using machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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
#_extreme is the mean of the 3 largest (most abnormal) values of that measurement across all nuclei in the sample.

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



# Correlation heatmap (mean features only); shows how strong correlation is between each feature with benign/malignant diagnosis as well as with each other
mean_features = [f for f in feature_names if f.endswith('_mean')] #extract mean features from feature list
corr = df[mean_features + ['diagnosis']].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
ax.set_title('Correlation Matrix (Mean Features + Diagnosis)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# preprocessing the data for modeling
X = df.drop('diagnosis', axis=1) # drop diagnosis column for feature df
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #split data into train and test sets (stratify =y makes sure the class distribution is preserved in both sets to avoid skew in results)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# feature engineering: PCA + SelectKBest

# PCA: reduce to top 10 components (captures most variance while reducing dimensionality)
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"  Cumulative PCA variance: {pca.explained_variance_ratio_.sum():.3f}")
# each component is a linear combination of the original features, ordered so that component 1 captures the most variance, component 2 the next most, etc.

# SelectKBest: keep top 15 features by ANOVA F-score
selector = SelectKBest(f_classif, k=15)
X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
X_test_kbest  = selector.transform(X_test_scaled)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"\n  Top 15 Features via SelectKBest:")
print("  " + ", ".join(selected_features))

# bundle feature sets for iteration
feature_sets = {
    'Full':    (X_train_scaled, X_test_scaled),
    'PCA':     (X_train_pca,    X_test_pca),
    'KBest':   (X_train_kbest,  X_test_kbest),
}

# train and evaluate multiple models (with hyperparameter tuning for main ones)

# hyperparameter tuning via GridSearchCV (on training set) 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nHyperparameter Tuning (GridSearchCV, 5-fold CV on Full feature set)")

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

# base model definitions (best tuned params from full-set grid search) 
base_models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'Random Forest':       rf_grid.best_estimator_,
    'SVM':                 svm_grid.best_estimator_,
    'Gradient Boosting':   gb_grid.best_estimator_,
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
}

#  train and evaluate every model on every feature set
# results dict: results[feature_set][model_name]
results = {fs: {} for fs in feature_sets}

for fs_name, (X_tr, X_te) in feature_sets.items():
    print(f"\nTraining on feature set: {fs_name}")
    for name, base_model in base_models.items():
        model = clone(base_model)  # fresh copy so fits don't bleed across feature sets
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='recall')
        results[fs_name][name] = {
            'model':          model,
            'y_pred':         y_pred,
            'accuracy':       accuracy_score(y_test, y_pred),
            'precision':      precision_score(y_test, y_pred),
            'recall':         recall_score(y_test, y_pred),
            'f1':             f1_score(y_test, y_pred),
            'cv_recall_mean': cv_scores.mean(),
            'cv_recall_std':  cv_scores.std(),
        }
        print(f"  {name:<22} recall={results[fs_name][name]['recall']:.3f}  "
              f"cv_recall={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

# evaluate all models

for fs_name in feature_sets:
    print(f"\nModel Performance — {fs_name} ")
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'CV Recall':>12}")
    print("-" * 76)
    for name, r in results[fs_name].items():
        print(f"{name:<22} {r['accuracy']:>10.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>10.3f} {r['f1']:>8.3f} "
              f"  {r['cv_recall_mean']:.3f}±{r['cv_recall_std']:.3f}")

model_names = list(base_models.keys())
fs_names    = list(feature_sets.keys())

# visualize model results
# confusion matrices (2 rows × 3 cols for 6 models)
fig, axes = plt.subplots(len(fs_names), len(model_names),
                         figsize=(24, 12))
for row, fs_name in enumerate(fs_names):
    for col, name in enumerate(model_names):
        ax = axes[row][col]
        r  = results[fs_name][name]
        cm = confusion_matrix(y_test, r['y_pred'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred. Benign', 'Pred. Malignant'], fontsize=7)
        ax.set_yticklabels(['Actual Benign', 'Actual Malignant'], fontsize=7)
        # ax.set_xlabel('Predicted', fontsize=7); ax.set_ylabel('Actual', fontsize=7)
        ax.set_title(f'{name}\n{fs_name} feat set Acc.={r["accuracy"]:.2f} Rec.={r["recall"]:.2f}',
                     fontsize=8)
        for rr in range(2):
            for cc in range(2):
                ax.text(cc, rr, str(cm[rr, cc]),
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color='white' if cm[rr, cc] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()


# CV Recall comparison: grouped bar chart (models × feature sets)
fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(model_names))
width  = 0.25
fs_colors = ['steelblue', 'darkorange', 'green']
for i, (fs_name, color) in enumerate(zip(fs_names, fs_colors)):
    means = [results[fs_name][n]['cv_recall_mean'] for n in model_names]
    stds  = [results[fs_name][n]['cv_recall_std']  for n in model_names]
    bars  = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                   label=fs_name, color=color, alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')
ax.set_ylim([0.82, 1.02])
ax.set_ylabel('CV Recall (mean ± std)')
ax.set_title('5-Fold CV Recall — All Models × Feature Sets')
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend(title='Feature Set')
plt.tight_layout()
plt.savefig('cv_recall_comparison.png', dpi=150)
plt.show()

# random forest feature importance (only doing for RF since it's the most interpretable model and has built-in feature importance)

rf_model = results['Full']['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
top20 = importances.sort_values(ascending=False).head(20)
bottomImportance = importances.sort_values(ascending=True).head(1)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue', ax=ax)
ax.set_title('Top 20 Feature Importances (Random Forest — Full)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

#  summary statistics and insights

# Find best model+feature set combo by CV recall
best_fs, best_name, best_cv = None, None, -1
for fs_name in feature_sets:
    for name in base_models:
        cv_mean = results[fs_name][name]['cv_recall_mean']
        if cv_mean > best_cv:
            best_cv, best_fs, best_name = cv_mean, fs_name, name

r = results[best_fs][best_name]
print(f"\nBest Model by CV Recall")
print(f"  {best_name} [{best_fs}]")
print(f"  Accuracy : {r['accuracy']:.3f}")
print(f"  Precision: {r['precision']:.3f}")
print(f"  Recall   : {r['recall']:.3f}")
print(f"  F1       : {r['f1']:.3f}")
print(f"  CV Recall: {r['cv_recall_mean']:.3f} ± {r['cv_recall_std']:.3f}")

print("\nTop 5 most important features (Random Forest — Full):")
for feat, score in importances.sort_values(ascending=False).head(5).items():
    print(f"  {feat:<30} {score:.4f}")

print(f"\nLeast important feature: {bottomImportance.index[0]}")


