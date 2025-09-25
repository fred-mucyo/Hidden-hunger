# predictive_model_smote_tuning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load dataset
# -----------------------------
data_path = "outputs/malnutrition_for_modeling.csv"
df = pd.read_csv(data_path)
print("Dataset preview:")
print(df.head(), "\n")

# -----------------------------
# 2. Select features and target
# -----------------------------
feature_cols = ["Underweight_pct", "Wasted_pct", "VitaminA_pct", "Iodine_pct"]
target_col = "high_risk_stunted"

X = df[feature_cols]
y = df[target_col]

# -----------------------------
# 3. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------------
# 4. Handle class imbalance with SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE, training class distribution:\n{pd.Series(y_train_res).value_counts()}")

# -----------------------------
# 5. Hyperparameter tuning with GridSearchCV
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_res, y_train_res)
best_rf = grid_search.best_estimator_
print(f"\nBest Random Forest parameters: {grid_search.best_params_}")

# -----------------------------
# 6. Evaluate tuned model
# -----------------------------
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.3f}")

# -----------------------------
# 7. Feature importance
# -----------------------------
feat_imp = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance - High Risk Stunted")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importance_smote.png", bbox_inches="tight")
plt.show()

# -----------------------------
# 8. Save predictions
# -----------------------------
df_preds = X_test.copy()
df_preds["true_label"] = y_test
df_preds["pred_label"] = y_pred
df_preds["pred_prob"] = y_prob
df_preds.to_csv("outputs/predictions_smote.csv", index=False)
print("\n✅ Predictions saved to outputs/predictions_smote.csv")
