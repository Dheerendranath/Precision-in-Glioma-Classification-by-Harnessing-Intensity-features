#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score


# Load data
file_path = '/Users/dheerendranathbattalapalli/Desktop/539_ANN_results.xlsx'
data = pd.read_excel(file_path)

# Prepare data
y = data.iloc[:, 0].values
X = data.iloc[:, 1:]

# Data preprocessing
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# PCA Analysis
pca = PCA()
pca.fit(X)

# Visualizing the PCA loadings for the features
loadings = pca.components_.T

plt.figure(figsize=(15, 7))

# Features in original space with redundant ones circled
plt.subplot(1, 2, 1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(X.columns)))

# Define a threshold to identify redundant features based on their loadings magnitude
threshold = 0.05
redundant_features = np.where(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2) < threshold)[0]

# Plotting the features with unique colors
for i, color in enumerate(colors):
    marker = '*' if i in redundant_features else 'o'
    plt.scatter(loadings[i, 0], loadings[i, 1], color=color, marker=marker)

plt.xlabel('Principal Component 1 Loadings')
plt.ylabel('Principal Component 2 Loadings')
plt.title('PCA Loadings')

# After excluding redundant features
selected_features = [feature for idx, feature in enumerate(X.columns) if idx not in redundant_features]
X_selected = X[selected_features]

pca_selected = PCA(n_components=2)
reduced_data = pca_selected.fit_transform(X_selected)

plt.subplot(1, 2, 2)
for idx, color in enumerate(colors):
    if idx not in redundant_features:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color=color, alpha=0.5)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Reduced data after removing redundant features')
plt.tight_layout()
plt.show()

# Exclude redundant features based on their low loadings magnitude
selected_features = [feature for idx, feature in enumerate(X.columns) if idx not in redundant_features]
X_selected = X[selected_features]

# Feature selection using Pearson correlation
threshold_for_feature_selection = 0.15
perfect_correlation = 1.0
selected_features = []

correlations = {}
p_values = {}  # Added dictionary to store p-values
t_statistics = {}  # Added dictionary to store t-statistics

for column in X.columns:
    corr, p_val = pearsonr(X[column], y)
    correlations[column] = corr
    p_values[column] = p_val  # Storing the p-value
    n = len(X[column])
    t_stat = corr * np.sqrt(n-2) / np.sqrt(1-corr**2)  # Calculation for t-statistic
    t_statistics[column] = t_stat  # Storing the t-statistic
    
    if abs(corr) > threshold_for_feature_selection:
        selected_features.append(column)

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Feature': selected_features,
    'Correlation': [correlations[feature] for feature in selected_features],
    'P-value': [p_values[feature] for feature in selected_features],
    'T-statistic': [t_statistics[feature] for feature in selected_features]
})

# Display the results
print(results_df)

# Create a correlation matrix for selected features
correlation_matrix = X[selected_features].corr()

# Identify features with perfect correlation
features_to_remove = set()  # Using a set to avoid duplicates
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) == perfect_correlation:
            colname = correlation_matrix.columns[i]  # Removing the second feature in the correlated pair
            features_to_remove.add(colname)

# Remove perfectly correlated features
selected_features = [feature for feature in selected_features if feature not in features_to_remove]

# Heatmap for selected features after removal of perfectly correlated ones
selected_corr_matrix = X[selected_features].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(selected_corr_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap After Removal of Perfectly Correlated Features")
plt.show()

# Use SMOTE for data augmentation
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X[selected_features], y)
X_smote = pd.DataFrame(X_smote, columns=selected_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)


from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score

# Define a custom scorer for specificity
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

specificity_scorer = make_scorer(specificity, greater_is_better=True)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=10, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Refit the classifier with the best hyperparameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Import additional classifiers
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Other necessary imports
from sklearn.metrics import f1_score, confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score

# Extended hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
grid_search_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=10, n_jobs=-1, verbose=2)
grid_search_svm.fit(X_train, y_train)


# Hyperparameter tuning for AdaBoost
param_grid_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}
grid_search_ada = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_ada, cv=10, n_jobs=-1, verbose=2)
grid_search_ada.fit(X_train, y_train)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Training predictions and probabilities
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else [None] * len(y_train_pred)

    # Testing predictions and probabilities
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [None] * len(y_test_pred)

    # Metrics calculation
    metrics_train = {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred),
        'Recall': recall_score(y_train, y_train_pred),
        'F1 Score': f1_score(y_train, y_train_pred),
        'AUC': roc_auc_score(y_train, y_train_prob) if y_train_prob[0] is not None else None
    }

    metrics_test = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1 Score': f1_score(y_test, y_test_pred),
        'AUC': roc_auc_score(y_test, y_test_prob) if y_test_prob[0] is not None else None
    }

    return metrics_train, metrics_test


# Evaluate each classifier
metrics_rf_train, metrics_rf_test = evaluate_model(best_rf, X_train, y_train, X_test, y_test)
metrics_svm_train, metrics_svm_test = evaluate_model(grid_search_svm.best_estimator_, X_train, y_train, X_test, y_test)


# Create a DataFrame for the results - Training
results_train_df = pd.DataFrame({
    'Classifier': ['RandomForest', 'SVM'],
    'Accuracy': [metrics_rf_train['Accuracy'], metrics_svm_train['Accuracy']],
    'Precision': [metrics_rf_train['Precision'], metrics_svm_train['Precision']],
    'Recall': [metrics_rf_train['Recall'], metrics_svm_train['Recall']],
    'F1 Score': [metrics_rf_train['F1 Score'], metrics_svm_train['F1 Score']],
    'AUC': [metrics_rf_train['AUC'], metrics_svm_train['AUC']]
})

# Create a DataFrame for the results - Testing
results_test_df = pd.DataFrame({
    'Classifier': ['RandomForest', 'SVM'],
    'Accuracy': [metrics_rf_test['Accuracy'], metrics_svm_test['Accuracy']],
    'Precision': [metrics_rf_test['Precision'], metrics_svm_test['Precision']],
    'Recall': [metrics_rf_test['Recall'], metrics_svm_test['Recall']],
    'F1 Score': [metrics_rf_test['F1 Score'], metrics_svm_test['F1 Score']],
    'AUC': [metrics_rf_test['AUC'], metrics_svm_test['AUC']]
})

# Display the results

import shap

# Assuming 'best_rf' is your trained Random Forest model
# and 'X_train' is your training dataset

# Calculate SHAP values
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_train)

# Summarize the SHAP values for each feature to get the mean absolute value
shap_sum = np.abs(shap_values[0]).mean(axis=0)
importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['Feature', 'SHAP Importance']
importance_df = importance_df.sort_values('SHAP Importance', ascending=False)

# Get top 5 features
top_features = importance_df.head(5)['Feature'].values

# Filter SHAP values for top 5 features
top_shap_values = shap_values[0][:, [X_train.columns.get_loc(feature) for feature in top_features]]

# Create SHAP summary plot for top 5 features
shap.summary_plot(top_shap_values, features=X_train[top_features], feature_names=top_features)

y_pred_svm = grid_search_svm.best_estimator_.predict(X_test)
y_prob_svm = grid_search_svm.best_estimator_.predict_proba(X_test)[:, 1]

# Confusion Matrices
cm_rf = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_rf, annot=True)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True)
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curves
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_prob_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




#print("Training Metrics:\n", results_train_df)
print("\nTesting Metrics:\n", results_test_df)

