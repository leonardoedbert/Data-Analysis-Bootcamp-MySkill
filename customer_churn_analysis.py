import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL PALETTE
# ============================================================
PALETTE   = ["#2D6A4F", "#52B788", "#B7E4C7", "#F4A261", "#E76F51",
              "#264653", "#457B9D", "#A8DADC", "#E9C46A", "#F77F00"]
PRIMARY   = "#2D6A4F"
SECONDARY = "#52B788"
ACCENT    = "#F4A261"
DANGER    = "#E76F51"
DARK      = "#264653"
LIGHT_BG  = "#F8F9FA"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": LIGHT_BG,
    "axes.facecolor":   LIGHT_BG,
    "axes.edgecolor":   DARK,
    "axes.labelcolor":  DARK,
    "text.color":       DARK,
    "xtick.color":      DARK,
    "ytick.color":      DARK,
    "grid.color":       "#DEE2E6",
    "font.family":      "DejaVu Sans",
})

def save(fig, name, dpi=180):
    fig.savefig(f"/home/claude/{name}", dpi=dpi, bbox_inches="tight",
                facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"  ✓ Saved {name}")

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("STEP 1 — DATA LOADING")
print("=" * 60)

df_raw = pd.read_csv('/mnt/user-data/uploads/ecommerce_customer_data_large.csv')
print(f"  Shape  : {df_raw.shape}")
print(f"  Columns: {list(df_raw.columns)}")

# ============================================================
# 2. DATA CLEANSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 — DATA CLEANSING")
print("=" * 60)

df = df_raw.copy()

# Fix duplicate age columns (Customer Age == Age — verified)
df.drop(columns=["Age"], inplace=True)

# Parse datetime
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])

# Fill missing Returns with 0 (no return info = no return)
missing_returns = df["Returns"].isna().sum()
df["Returns"] = df["Returns"].fillna(0).astype(int)
print(f"  Filled {missing_returns} missing 'Returns' → 0")

# Drop non-informative columns for ML
df.drop(columns=["Customer Name"], inplace=True)

print(f"  Remaining nulls: {df.isnull().sum().sum()}")
print(f"  Final shape    : {df.shape}")

# ── Plot: Missing value summary ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Data Cleansing Overview", fontsize=15, fontweight="bold", color=DARK)

missing_before = df_raw.isnull().sum()
missing_before = missing_before[missing_before > 0]
axes[0].barh(missing_before.index, missing_before.values, color=DANGER, edgecolor=DARK)
axes[0].set_title("Missing Values Before Cleaning", fontweight="bold")
axes[0].set_xlabel("Count")
for i, v in enumerate(missing_before.values):
    axes[0].text(v + 300, i, f"{v:,}", va="center", fontsize=10, color=DARK)

dtypes_count = df.dtypes.astype(str).value_counts()
axes[1].bar(dtypes_count.index, dtypes_count.values,
            color=[PRIMARY, SECONDARY, ACCENT][:len(dtypes_count)], edgecolor=DARK)
axes[1].set_title("Data Types After Cleaning", fontweight="bold")
axes[1].set_xlabel("dtype")
axes[1].set_ylabel("Number of Columns")

plt.tight_layout()
save(fig, "01_data_cleansing.png")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

# Customer-level aggregation (each row = one transaction)
cust = (df.groupby("Customer ID")
          .agg(
              total_spend        = ("Total Purchase Amount", "sum"),
              avg_spend          = ("Total Purchase Amount", "mean"),
              num_transactions   = ("Total Purchase Amount", "count"),
              avg_price          = ("Product Price", "mean"),
              total_quantity     = ("Quantity", "sum"),
              total_returns      = ("Returns", "sum"),
              return_rate        = ("Returns", "mean"),
              age                = ("Customer Age", "first"),
              gender             = ("Gender", "first"),
              churn              = ("Churn", "first"),
              fav_category       = ("Product Category", lambda x: x.mode()[0]),
              fav_payment        = ("Payment Method", lambda x: x.mode()[0]),
              last_purchase_date = ("Purchase Date", "max"),
              first_purchase_date= ("Purchase Date", "min"),
          ).reset_index()
       )

# Recency (days since last purchase relative to dataset max date)
max_date = df["Purchase Date"].max()
cust["recency_days"] = (max_date - cust["last_purchase_date"]).dt.days
cust["tenure_days"]  = (cust["last_purchase_date"] - cust["first_purchase_date"]).dt.days
cust["avg_days_between"] = np.where(
    cust["num_transactions"] > 1,
    cust["tenure_days"] / (cust["num_transactions"] - 1),
    cust["tenure_days"]
)

# Age bins
cust["age_group"] = pd.cut(cust["age"], bins=[17,30,45,60,71],
                            labels=["18-30","31-45","46-60","61-70"])

# Drop raw date cols
cust.drop(columns=["last_purchase_date","first_purchase_date"], inplace=True)

print(f"  Customer-level dataset shape: {cust.shape}")
print(f"  Churn rate: {cust['churn'].mean():.2%}")

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Plot A: Target distribution ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Target Variable — Customer Churn", fontsize=15, fontweight="bold", color=DARK)

churn_counts = cust["churn"].value_counts()
labels = ["Tidak Churn (0)", "Churn (1)"]
colors_pie = [SECONDARY, DANGER]
axes[0].pie(churn_counts.values, labels=labels, colors=colors_pie,
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor=LIGHT_BG, linewidth=2))
axes[0].set_title("Distribusi Churn", fontweight="bold")

churn_by_gender = cust.groupby(["gender","churn"]).size().unstack()
churn_by_gender.plot(kind="bar", ax=axes[1], color=[SECONDARY, DANGER],
                     edgecolor=DARK, width=0.6)
axes[1].set_title("Churn by Gender", fontweight="bold")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Jumlah Pelanggan")
axes[1].legend(["Tidak Churn","Churn"], framealpha=0.7)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
save(fig, "02_target_distribution.png")

# ── Plot B: Numerical distributions ─────────────────────────
num_cols = ["total_spend","avg_spend","num_transactions",
            "return_rate","recency_days","tenure_days"]
labels_map = {
    "total_spend":"Total Spend",
    "avg_spend":"Avg Spend/Transaksi",
    "num_transactions":"Jumlah Transaksi",
    "return_rate":"Return Rate",
    "recency_days":"Recency (hari)",
    "tenure_days":"Tenure (hari)"
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribusi Fitur Numerik (Churn vs Non-Churn)", fontsize=15,
             fontweight="bold", color=DARK)
axes = axes.flatten()

for i, col in enumerate(num_cols):
    for label, color in zip([0,1],[SECONDARY, DANGER]):
        subset = cust[cust["churn"]==label][col]
        axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                     edgecolor="none", label=["Tidak Churn","Churn"][label],
                     density=True)
    axes[i].set_title(labels_map[col], fontweight="bold")
    axes[i].set_xlabel(labels_map[col])
    axes[i].set_ylabel("Density")
    axes[i].legend(fontsize=9)

plt.tight_layout()
save(fig, "03_numeric_distributions.png")

# ── Plot C: Categorical analysis ─────────────────────────────
cat_cols = ["fav_category","fav_payment","age_group"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Churn Rate per Kategori", fontsize=15, fontweight="bold", color=DARK)

for i, col in enumerate(cat_cols):
    churn_rate = cust.groupby(col)["churn"].mean().sort_values(ascending=False)
    bars = axes[i].bar(churn_rate.index.astype(str), churn_rate.values,
                       color=PALETTE[:len(churn_rate)], edgecolor=DARK, width=0.6)
    axes[i].set_title(col.replace("_"," ").title(), fontweight="bold")
    axes[i].set_ylabel("Churn Rate")
    axes[i].set_ylim(0, churn_rate.max() * 1.3)
    for bar, val in zip(bars, churn_rate.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
save(fig, "04_categorical_churn.png")

# ── Plot D: Correlation heatmap ──────────────────────────────
num_features = ["total_spend","avg_spend","num_transactions","avg_price",
                "total_quantity","total_returns","return_rate","recency_days",
                "tenure_days","age","churn"]
corr = cust[num_features].corr()

fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle("Correlation Matrix", fontsize=15, fontweight="bold", color=DARK)
cmap = sns.diverging_palette(145, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size":9})
ax.set_title("")
plt.tight_layout()
save(fig, "05_correlation_heatmap.png")

# ── Plot E: RFM-style scatterplot ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RFM Analysis — Recency vs Spend & Frequency", fontsize=15,
             fontweight="bold", color=DARK)

for ax, (y_col, y_lbl) in zip(axes, [("total_spend","Total Spend"),
                                      ("num_transactions","Jumlah Transaksi")]):
    sample = cust.sample(min(3000, len(cust)), random_state=42)
    colors_map = sample["churn"].map({0: SECONDARY, 1: DANGER})
    ax.scatter(sample["recency_days"], sample[y_col], c=colors_map,
               alpha=0.35, s=18, edgecolors="none")
    ax.set_xlabel("Recency (hari)", fontweight="bold")
    ax.set_ylabel(y_lbl, fontweight="bold")
    ax.set_title(f"Recency vs {y_lbl}", fontweight="bold")
    # legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SECONDARY, label="Tidak Churn"),
                        Patch(color=DANGER, label="Churn")],
              framealpha=0.8)

plt.tight_layout()
save(fig, "06_rfm_scatter.png")

# ============================================================
# 5. MODELLING PREPARATION (Split & Encode)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 — TRAIN / TEST SPLIT")
print("=" * 60)

feature_cols = ["total_spend","avg_spend","num_transactions","avg_price",
                "total_quantity","total_returns","return_rate","recency_days",
                "tenure_days","avg_days_between","age",
                "gender","fav_category","fav_payment"]

X = cust[feature_cols].copy()
y = cust["churn"]

# Encode categoricals
le = LabelEncoder()
for col in ["gender","fav_category","fav_payment"]:
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
print(f"  Train churn rate: {y_train.mean():.2%}")
print(f"  Test  churn rate: {y_test.mean():.2%}")

# ── Plot: Split viz ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Train / Test Split", fontsize=15, fontweight="bold", color=DARK)

sizes = [len(X_train), len(X_test)]
axes[0].bar(["Train","Test"], sizes, color=[PRIMARY, ACCENT], edgecolor=DARK, width=0.5)
axes[0].set_title("Jumlah Data per Set", fontweight="bold")
axes[0].set_ylabel("Jumlah Pelanggan")
for i, v in enumerate(sizes):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

for ax, (data, label) in zip([axes[1]], [([y_train, y_test], ["Train","Test"])]):
    rates = [d.mean() for d in data]
    bars = ax.bar(label, rates, color=[PRIMARY, ACCENT], edgecolor=DARK, width=0.5)
    ax.set_title("Churn Rate per Set", fontweight="bold")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, max(rates)*1.3)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{r:.2%}", ha="center", fontweight="bold")

plt.tight_layout()
save(fig, "07_train_test_split.png")

# ============================================================
# 6. MODELLING
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 — MODELLING")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                                   class_weight="balanced", n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                       max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    X_use = X_train_sc if name == "Logistic Regression" else X_train.values
    cv_auc = cross_val_score(model, X_use, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_use, y_train)

    X_te = X_test_sc if name == "Logistic Regression" else X_test.values
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)
    test_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model":     model,
        "y_prob":    y_prob,
        "y_pred":    y_pred,
        "cv_auc":    cv_auc,
        "cv_mean":   cv_auc.mean(),
        "cv_std":    cv_auc.std(),
        "test_auc":  test_auc,
        "report":    classification_report(y_test, y_pred, output_dict=True),
    }
    print(f"  {name:<24} CV-AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}  Test-AUC={test_auc:.4f}")

# ── Plot: Model comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Perbandingan Model", fontsize=15, fontweight="bold", color=DARK)

model_names = list(results.keys())
cv_means    = [results[m]["cv_mean"]  for m in model_names]
cv_stds     = [results[m]["cv_std"]   for m in model_names]
test_aucs   = [results[m]["test_auc"] for m in model_names]

x = np.arange(len(model_names))
bars = axes[0].bar(x, cv_means, yerr=cv_stds, color=PALETTE[:4], edgecolor=DARK,
                   capsize=5, width=0.6)
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, rotation=15, ha="right")
axes[0].set_title("CV AUC-ROC (5-Fold)", fontweight="bold")
axes[0].set_ylabel("AUC-ROC")
axes[0].set_ylim(0.5, 1.0)
for bar, val in zip(bars, cv_means):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

# F1-precision-recall bar
metrics_data = {}
for m in model_names:
    rep = results[m]["report"]
    metrics_data[m] = {
        "Precision": rep["1"]["precision"],
        "Recall":    rep["1"]["recall"],
        "F1":        rep["1"]["f1-score"],
    }

x = np.arange(len(model_names))
width = 0.25
metric_colors = [PRIMARY, SECONDARY, ACCENT]
for i, (metric, color) in enumerate(zip(["Precision","Recall","F1"], metric_colors)):
    vals = [metrics_data[m][metric] for m in model_names]
    axes[1].bar(x + i*width, vals, width, label=metric, color=color, edgecolor=DARK)

axes[1].set_xticks(x + width); axes[1].set_xticklabels(model_names, rotation=15, ha="right")
axes[1].set_title("Precision / Recall / F1 (Class Churn)", fontweight="bold")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1.1)
axes[1].legend()

plt.tight_layout()
save(fig, "08_model_comparison.png")

# ============================================================
# 7. EVALUATION — Best Model Deep-Dive
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 — EVALUATION")
print("=" * 60)

best_name = max(results, key=lambda m: results[m]["test_auc"])
best      = results[best_name]
print(f"  Best model: {best_name}  (Test AUC = {best['test_auc']:.4f})")

# ── Plot: ROC all models ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Evaluation Curves", fontsize=15, fontweight="bold", color=DARK)

for (name, res), color in zip(results.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC={res['test_auc']:.3f})")
axes[0].plot([0,1],[0,1],"--", color="#ADB5BD", lw=1.2)
axes[0].fill_between(*roc_curve(y_test, best["y_prob"])[:2],
                      alpha=0.08, color=PRIMARY)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Semua Model", fontweight="bold")
axes[0].legend(fontsize=8, loc="lower right")

# Precision-Recall
for (name, res), color in zip(results.items(), PALETTE):
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ap = average_precision_score(y_test, res["y_prob"])
    axes[1].plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
axes[1].axhline(y=y_test.mean(), color="#ADB5BD", linestyle="--", lw=1.2,
                label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve", fontweight="bold")
axes[1].legend(fontsize=8)

plt.tight_layout()
save(fig, "09_roc_pr_curves.png")

# ── Plot: Confusion matrices ─────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Confusion Matrix — Semua Model", fontsize=15, fontweight="bold", color=DARK)

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    cmap_cm = sns.light_palette(PRIMARY, as_cmap=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Tidak Churn","Churn"])
    disp.plot(ax=ax, cmap=cmap_cm, colorbar=False)
    ax.set_title(name, fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
save(fig, "10_confusion_matrices.png")

# ── Plot: Feature Importance (best model) ───────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle(f"Feature Importance — {best_name}", fontsize=15,
             fontweight="bold", color=DARK)

best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
    imp.plot(kind="barh", ax=ax, color=colors_bar, edgecolor=DARK)
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_title("")
    for i, v in enumerate(imp.values):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8)
else:
    coef = pd.Series(np.abs(best_model.coef_[0]), index=feature_cols).sort_values(ascending=True)
    coef.plot(kind="barh", ax=ax, color=PRIMARY, edgecolor=DARK)
    ax.set_xlabel("|Coefficient|", fontweight="bold")

plt.tight_layout()
save(fig, "11_feature_importance.png")

# ── Plot: Score distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Distribusi Skor Prediksi — {best_name}", fontsize=15,
             fontweight="bold", color=DARK)

for label, color in zip([0,1],[SECONDARY, DANGER]):
    mask = y_test == label
    axes[0].hist(best["y_prob"][mask], bins=40, alpha=0.65, color=color,
                 label=["Tidak Churn","Churn"][label], density=True, edgecolor="none")
axes[0].set_xlabel("Probabilitas Churn"); axes[0].set_ylabel("Density")
axes[0].set_title("Distribusi Prob. per Kelas", fontweight="bold")
axes[0].legend()
axes[0].axvline(0.5, color=DARK, linestyle="--", lw=1.5, label="Threshold 0.5")

# Calibration-like: avg pred prob per decile
df_cal = pd.DataFrame({"prob": best["y_prob"], "actual": y_test.values})
df_cal["decile"] = pd.qcut(df_cal["prob"], 10, labels=False, duplicates="drop")
cal_agg = df_cal.groupby("decile").agg(avg_pred=("prob","mean"),
                                        avg_actual=("actual","mean")).reset_index()
axes[1].plot(cal_agg["avg_pred"], cal_agg["avg_actual"], "o-",
             color=PRIMARY, lw=2, markersize=7, label="Model")
axes[1].plot([0,1],[0,1],"--", color=DANGER, lw=1.5, label="Perfect Calibration")
axes[1].set_xlabel("Rata-rata Prob. Prediksi"); axes[1].set_ylabel("Rata-rata Aktual")
axes[1].set_title("Reliability Diagram (Calibration)", fontweight="bold")
axes[1].legend()

plt.tight_layout()
save(fig, "12_score_distribution.png")

# ── Summary table ────────────────────────────────────────────
print("\n📊 FINAL SUMMARY TABLE")
print("-" * 70)
print(f"{'Model':<26} {'CV-AUC':>10} {'Test-AUC':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 70)
for name, res in results.items():
    rep = res["report"]["1"]
    marker = " ◀ BEST" if name == best_name else ""
    print(f"{name:<26} {res['cv_mean']:>10.4f} {res['test_auc']:>10.4f} "
          f"{rep['precision']:>10.4f} {rep['recall']:>8.4f} {rep['f1-score']:>8.4f}{marker}")
print("-" * 70)
print("\n✅ All plots saved to /home/claude/")
