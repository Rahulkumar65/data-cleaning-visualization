"""
Predictive Modeling Using Machine Learning
==========================================
Author  : [Your Name]
Due Date: 05 May 2026
Dataset : Pima Indians Diabetes Dataset (768 records, 8 features)
Task    : Binary classification — predict diabetes (0 = No, 1 = Yes)

Algorithms : Logistic Regression, Decision Tree, Random Forest
Libraries  : pandas, numpy, scikit-learn, matplotlib, seaborn

Run: python predictive_modeling_ml.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                      "axes.spines.right": False})

# ─────────────────────────────────────────────
# 1. LOAD & INSPECT DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Load & Inspect Dataset")
print("=" * 60)

# The Pima Indians Diabetes dataset columns
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Generate a realistic synthetic version of the Pima dataset
np.random.seed(42)
N = 768

data = pd.DataFrame({
    "Pregnancies"              : np.random.randint(0, 17, N),
    "Glucose"                  : np.random.normal(121, 32, N).clip(44, 199).astype(int),
    "BloodPressure"            : np.random.normal(69, 19, N).clip(24, 122).astype(int),
    "SkinThickness"            : np.random.normal(21, 16, N).clip(7, 99).astype(int),
    "Insulin"                  : np.abs(np.random.normal(80, 115, N)).clip(14, 846).astype(int),
    "BMI"                      : np.round(np.random.normal(32, 7.9, N).clip(18, 67), 1),
    "DiabetesPedigreeFunction" : np.round(np.abs(np.random.normal(0.47, 0.33, N)).clip(0.08, 2.42), 3),
    "Age"                      : np.random.randint(21, 81, N),
})

# Outcome correlated with glucose & BMI
prob = 1 / (1 + np.exp(-(
    -6.5
    + 0.04  * data["Glucose"]
    + 0.09  * data["BMI"]
    + 0.03  * data["Age"]
    + 0.5   * data["DiabetesPedigreeFunction"]
)))
data["Outcome"] = (np.random.rand(N) < prob).astype(int)

# Inject zeros (missing in original dataset convention)
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    zero_idx = np.random.choice(N, int(N * 0.05), replace=False)
    data.loc[zero_idx, col] = 0

print(f"  Shape        : {data.shape}")
print(f"  Columns      : {list(data.columns)}")
print(f"  Class balance: {data['Outcome'].value_counts().to_dict()}")
print(f"  Zeros (missing markers):\n{(data == 0).sum()}")
print()

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 2 — Data Preprocessing")
print("=" * 60)

df = data.copy()

# Replace 0s with NaN in physiological columns (0 is biologically impossible)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
print(f"  Zeros replaced with NaN in: {zero_cols}")

# Impute with median (robust to outliers)
for col in zero_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"    {col:30s} → imputed with median ({median_val:.1f})")

print(f"\n  Remaining nulls : {df.isnull().sum().sum()}")
print()

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 3 — Feature Engineering")
print("=" * 60)

df["GlucoseBMI"]    = df["Glucose"] * df["BMI"]
df["AgePedigree"]   = df["Age"] * df["DiabetesPedigreeFunction"]
df["InsulinLog"]    = np.log1p(df["Insulin"])

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
            "GlucoseBMI", "AgePedigree", "InsulinLog"]

X = df[features]
y = df["Outcome"]
print(f"  Features used : {features}")
print(f"  X shape       : {X.shape}")
print()

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 4 — Train/Test Split & Scaling")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train size : {X_train.shape[0]} samples")
print(f"  Test size  : {X_test.shape[0]} samples")
print(f"  Features   : {X_train.shape[1]}")
print(f"  Scaled using StandardScaler")
print()

# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 5 — Training Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Use scaled data for Logistic Regression, raw for tree models
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    Xall = np.vstack([Xtr, Xte]) if name == "Logistic Regression" else X.values

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
        "auc"      : roc_auc_score(y_test, y_prob),
        "cv_mean"  : cv_scores.mean(),
        "cv_std"   : cv_scores.std(),
        "cm"       : confusion_matrix(y_test, y_pred),
    }
    print(f"  [{name}]")
    print(f"    Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"    Precision : {results[name]['precision']:.4f}")
    print(f"    Recall    : {results[name]['recall']:.4f}")
    print(f"    F1-Score  : {results[name]['f1']:.4f}")
    print(f"    ROC-AUC   : {results[name]['auc']:.4f}")
    print(f"    CV Score  : {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}")
    print()

best_name = max(results, key=lambda k: results[k]["auc"])
print(f"  Best model (by AUC): {best_name}")
print()

# ─────────────────────────────────────────────
# 6. CLASSIFICATION REPORTS
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 6 — Classification Reports")
print("=" * 60)
for name, r in results.items():
    print(f"\n  --- {name} ---")
    print(classification_report(y_test, r["y_pred"],
                                target_names=["No Diabetes", "Diabetes"]))

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 7 — Generating Visualizations")
print("=" * 60)

COLORS = {"Logistic Regression": "#EF9F27",
          "Decision Tree"      : "#378ADD",
          "Random Forest"      : "#1D9E75"}

# ── Fig 1: Full dashboard ────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Predictive Modeling — ML Report (Diabetes Dataset)",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Model accuracy comparison
ax1 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
colors = [COLORS[n] for n in names]
bars = ax1.bar(range(len(names)), accs, color=colors, edgecolor="none", width=0.5)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(60, 95)
ax1.set_title("Model Accuracy Comparison", fontsize=11, pad=8)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

# 1b. ROC Curves
ax2 = fig.add_subplot(gs[0, 1])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax2.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})",
             color=COLORS[name], linewidth=2)
ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves", fontsize=11, pad=8)
ax2.legend(fontsize=8, loc="lower right")

# 1c. Confusion matrix — best model
ax3 = fig.add_subplot(gs[0, 2])
cm = results[best_name]["cm"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax3,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            cbar_kws={"shrink": 0.8}, linewidths=0.5)
ax3.set_title(f"Confusion Matrix — {best_name}", fontsize=11, pad=8)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# 1d. Feature importance (Random Forest)
ax4 = fig.add_subplot(gs[1, 0])
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()
importances.plot(kind="barh", ax=ax4, color="#7F77DD", edgecolor="none")
ax4.set_title("Feature Importance — Random Forest", fontsize=11, pad=8)
ax4.set_xlabel("Importance Score")

# 1e. Cross-validation scores
ax5 = fig.add_subplot(gs[1, 1])
cv_means = [results[n]["cv_mean"] * 100 for n in names]
cv_stds  = [results[n]["cv_std"] * 100 for n in names]
ax5.bar(range(len(names)), cv_means, yerr=cv_stds, color=colors,
        edgecolor="none", width=0.5, capsize=6, error_kw={"linewidth": 1.5})
ax5.set_xticks(range(len(names)))
ax5.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
ax5.set_ylabel("CV Accuracy (%)")
ax5.set_ylim(60, 95)
ax5.set_title("5-Fold Cross-Validation Scores", fontsize=11, pad=8)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax5.text(i, m + s + 0.5, f"{m:.1f}%", ha="center", fontsize=9)

# 1f. Precision / Recall / F1 grouped bar
ax6 = fig.add_subplot(gs[1, 2])
metrics = ["Precision", "Recall", "F1"]
x = np.arange(len(metrics))
w = 0.25
for i, name in enumerate(names):
    vals = [results[name]["precision"],
            results[name]["recall"],
            results[name]["f1"]]
    ax6.bar(x + i * w, [v * 100 for v in vals], w,
            label=name, color=COLORS[name], edgecolor="none")
ax6.set_xticks(x + w)
ax6.set_xticklabels(metrics)
ax6.set_ylabel("Score (%)")
ax6.set_ylim(50, 100)
ax6.set_title("Precision / Recall / F1", fontsize=11, pad=8)
ax6.legend(fontsize=8)

plt.savefig("ml_dashboard.png", bbox_inches="tight", facecolor="white")
print("  [Saved] ml_dashboard.png")

# ── Fig 2: Decision Tree Visualization ──────
fig2, ax = plt.subplots(figsize=(20, 8))
plot_tree(results["Decision Tree"]["model"],
          feature_names=features,
          class_names=["No Diabetes", "Diabetes"],
          filled=True, rounded=True, fontsize=8, ax=ax,
          max_depth=3)
ax.set_title("Decision Tree Structure (max_depth=3 shown)", fontsize=13, pad=12)
plt.savefig("decision_tree.png", bbox_inches="tight", facecolor="white")
print("  [Saved] decision_tree.png")

# ── Fig 3: Correlation heatmap ───────────────
fig3, ax = plt.subplots(figsize=(10, 8))
corr = df[features + ["Outcome"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight", facecolor="white")
print("  [Saved] correlation_heatmap.png")

# ── Fig 4: Prediction probability distribution ─
fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
fig4.suptitle("Predicted Probability Distribution by True Class", fontsize=12)
for ax, (name, r) in zip(axes, results.items()):
    prob_no  = r["y_prob"][y_test == 0]
    prob_yes = r["y_prob"][y_test == 1]
    ax.hist(prob_no,  bins=20, alpha=0.7, color="#378ADD", label="No Diabetes", density=True)
    ax.hist(prob_yes, bins=20, alpha=0.7, color="#E24B4A", label="Diabetes",    density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("probability_distributions.png", bbox_inches="tight", facecolor="white")
print("  [Saved] probability_distributions.png")

# ─────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Best model    : {best_name}")
print(f"  Accuracy      : {results[best_name]['accuracy']*100:.2f}%")
print(f"  ROC-AUC       : {results[best_name]['auc']:.4f}")
print(f"  F1-Score      : {results[best_name]['f1']:.4f}")
print(f"  Top feature   : {importances.index[-1]} "
      f"({importances.values[-1]*100:.1f}% importance)")
print()
print("  Output files:")
print("    ml_dashboard.png")
print("    decision_tree.png")
print("    correlation_heatmap.png")
print("    probability_distributions.png")
print()
print("  All done! Open PNG files to view visualizations.")
