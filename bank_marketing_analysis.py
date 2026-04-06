# ============================================================
# BANK MARKETING — END-TO-END ANALYSIS
# Dataset: bank-full.csv (UCI Bank Marketing Dataset)
# Sections:
#   1. Data Loading
#   2. Data Cleaning
#   3. Feature Engineering
#   4. Exploratory Data Analysis (10 plots)
#   5. Preprocessing & Train-Test Split
#   6. Modeling (Logistic Regression, RF, GBM)
#   7. Evaluation (4 plots)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, average_precision_score,
    precision_recall_curve
)

# ─────────────────────────────────────────────────────────────
# GLOBAL PALETTE  (consistent across ALL plots)
# ─────────────────────────────────────────────────────────────
PALETTE   = ["#1b4332", "#2d6a4f", "#52b788", "#95d5b2", "#d8f3dc"]
NEG_COLOR = "#e76f51"
POS_COLOR = "#52b788"
PRIMARY   = "#1b4332"
SECONDARY = "#2d6a4f"
ACCENT    = "#95d5b2"
BG        = "#f8faf9"
GRID_C    = "#dee2e6"
FTITLE    = {"fontsize": 14, "fontweight": "bold", "color": PRIMARY}
FLABEL    = {"fontsize": 11, "color": "#343a40"}

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor"  : GRID_C, "grid.color": GRID_C,
    "grid.linewidth"  : 0.6,    "axes.grid": True,
    "font.family"     : "DejaVu Sans",
})

def style_ax(ax, title, xlabel="", ylabel="", legend=True):
    """Apply consistent styling to any Axes object."""
    ax.set_title(title, **FTITLE, pad=12)
    if xlabel: ax.set_xlabel(xlabel, **FLABEL)
    if ylabel: ax.set_ylabel(ylabel, **FLABEL)
    ax.tick_params(colors="#495057", labelsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    if legend and ax.get_legend(): ax.get_legend().set_title("")
    return ax

OUT = "./"   # folder to save plots — change as needed


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. DATA LOADING")
print("=" * 60)

df = pd.read_csv("bank-full.csv", sep=",")
print(f"Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns : {df.columns.tolist()}")
print(df.head())


# ═══════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. DATA CLEANING")
print("=" * 60)

# Replace 'unknown' strings with NaN
df.replace("unknown", np.nan, inplace=True)

# pdays = -1 means never contacted before → recode to 0, add flag
df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
df["pdays"] = df["pdays"].replace(-1, 0)

# Impute categorical NaNs with mode
for col in ["job", "education", "contact", "poutcome"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode binary target
df["y_bin"] = (df["y"] == "yes").astype(int)

print(f"After cleaning : {df.shape[0]:,} rows")
print(f"Remaining NaNs : {df.isnull().sum().sum()}")
print(f"Target balance — No: {(df.y_bin==0).sum():,} | Yes: {(df.y_bin==1).sum():,}")


# ═══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

# Month → numeric & quarter
month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
df["month_num"] = df["month"].map(month_map)
df["quarter"]   = pd.cut(df["month_num"], bins=[0,3,6,9,12],
                          labels=["Q1","Q2","Q3","Q4"])

# Age group
df["age_group"] = pd.cut(df["age"], bins=[17,25,35,45,55,95],
                          labels=["18-25","26-35","36-45","46-55","56+"])

# Balance transformations
df["balance_log"]  = np.log1p(df["balance"].clip(lower=0))
df["high_balance"] = (df["balance"] > df["balance"].quantile(0.75)).astype(int)

# Call engagement
df["contact_intensity"] = df["campaign"] * df["duration"]
df["duration_min"]      = df["duration"] / 60

# Combined loan flag
df["has_loan_or_housing"] = (
    (df["housing"] == "yes") | (df["loan"] == "yes")
).astype(int)

new_feats = ["month_num","quarter","age_group","balance_log",
             "high_balance","contact_intensity","duration_min",
             "has_loan_or_housing","was_contacted_before"]
print(f"New features: {new_feats}")


# ═══════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ─── PLOT 1: Target Distribution ─────────────────────────────
counts = df["y"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bars = axes[0].bar(["No (not subscribed)","Yes (subscribed)"],
                   counts.values, color=[NEG_COLOR, POS_COLOR],
                   edgecolor="white", width=0.5)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=11, color=PRIMARY, fontweight="bold")
style_ax(axes[0], "Term Deposit Subscription Count",
         ylabel="Number of Clients", legend=False)

wedges, texts, autos = axes[1].pie(
    counts.values, labels=["No","Yes"],
    autopct="%1.1f%%", startangle=90,
    colors=[NEG_COLOR, POS_COLOR], pctdistance=0.75,
    wedgeprops={"linewidth":2, "edgecolor":"white"})
for t in autos:
    t.set_fontsize(13); t.set_color("white"); t.set_fontweight("bold")
axes[1].add_patch(plt.Circle((0,0), 0.5, fc=BG))
axes[1].set_title("Subscription Ratio", **FTITLE)

plt.suptitle("Target Variable Distribution",
             fontsize=16, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 1 — Target Distribution")

# ─── PLOT 2: Age Distribution by Outcome ─────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for label, color, sub in [("No", NEG_COLOR, df[df.y=="no"]["age"]),
                           ("Yes", POS_COLOR, df[df.y=="yes"]["age"])]:
    ax.hist(sub, bins=35, alpha=0.65, color=color,
            label=f"{label} ({len(sub):,})", edgecolor="white", lw=0.4)
ax.axvline(df[df.y=="no"]["age"].mean(), color=NEG_COLOR, ls="--", lw=1.8,
           label=f"Mean(No)  = {df[df.y=='no']['age'].mean():.1f}")
ax.axvline(df[df.y=="yes"]["age"].mean(), color=POS_COLOR, ls="--", lw=1.8,
           label=f"Mean(Yes) = {df[df.y=='yes']['age'].mean():.1f}")
ax.legend(fontsize=10)
style_ax(ax, "Age Distribution by Subscription Outcome",
         xlabel="Age", ylabel="Count")
plt.tight_layout()
plt.savefig(f"{OUT}plot_02_age_distribution.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 2 — Age Distribution")

# ─── PLOT 3: Subscription Rate by Job ────────────────────────
job_rate = (df.groupby("job")["y_bin"]
              .agg(["mean","count"])
              .reset_index()
              .rename(columns={"mean":"sub_rate","count":"n"})
              .sort_values("sub_rate", ascending=True))
job_rate = job_rate[job_rate["n"] > 50]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(job_rate["job"], job_rate["sub_rate"]*100,
               color=[POS_COLOR if v > job_rate["sub_rate"].mean()
                      else PALETTE[1] for v in job_rate["sub_rate"]],
               edgecolor="white", height=0.6)
ax.axvline(job_rate["sub_rate"].mean()*100, color=ACCENT,
           ls="--", lw=1.8, label=f"Avg = {job_rate['sub_rate'].mean()*100:.1f}%")
for bar, (_, row) in zip(bars, job_rate.iterrows()):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f"{row['sub_rate']*100:.1f}%  (n={int(row['n']):,})",
            va="center", fontsize=9, color=PRIMARY)
ax.set_xlim(0, 45)
style_ax(ax, "Subscription Rate by Job Type",
         xlabel="Subscription Rate (%)", legend=False)
plt.tight_layout()
plt.savefig(f"{OUT}plot_03_subscription_by_job.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 3 — Subscription by Job")

# ─── PLOT 4: Monthly Campaign Volume & Subscription Rate ──────
monthly = (df.groupby("month_num")
             .agg(total=("y_bin","count"), subscribed=("y_bin","sum"))
             .reset_index())
monthly["rate"] = monthly["subscribed"] / monthly["total"] * 100
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_label"] = monthly["month_num"].apply(lambda x: month_labels[x-1])

fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
ax1.bar(monthly["month_label"], monthly["total"],
        color=ACCENT, alpha=0.7, label="Total Contacts", edgecolor="white")
ax2.plot(monthly["month_label"], monthly["rate"],
         color=PRIMARY, lw=2.5, marker="o", ms=7, label="Sub Rate %")
ax2.set_ylabel("Subscription Rate (%)", **FLABEL)
ax1.set_ylabel("Total Contacts", **FLABEL)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=10, loc="upper left")
style_ax(ax1, "Monthly Campaign Volume & Subscription Rate",
         xlabel="Month", legend=False)
for sp in ax2.spines.values(): sp.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}plot_04_monthly_campaign.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 4 — Monthly Campaign")

# ─── PLOT 5: Balance Distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for label, color, sub in [("No", NEG_COLOR, df[df.y=="no"]["balance"]),
                           ("Yes", POS_COLOR, df[df.y=="yes"]["balance"])]:
    axes[0].hist(sub.clip(-500, 10000), bins=50, alpha=0.6,
                 color=color, label=label, edgecolor="white", lw=0.3)
axes[0].legend(fontsize=10)
style_ax(axes[0], "Account Balance (Raw)", xlabel="Balance (EUR)", ylabel="Count")

for label, color, sub in [("No", NEG_COLOR, df[df.y=="no"]["balance_log"]),
                           ("Yes", POS_COLOR, df[df.y=="yes"]["balance_log"])]:
    axes[1].hist(sub, bins=50, alpha=0.6, color=color,
                 label=label, edgecolor="white", lw=0.3)
axes[1].legend(fontsize=10)
style_ax(axes[1], "Account Balance (Log-scaled)",
         xlabel="log(Balance + 1)", ylabel="Count")
plt.suptitle("Balance Distribution by Subscription Outcome",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_05_balance_distribution.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 5 — Balance Distribution")

# ─── PLOT 6: Call Duration Analysis ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="y", y="duration_min", order=["no","yes"],
            palette=[NEG_COLOR, POS_COLOR], width=0.45,
            linewidth=1.2, ax=axes[0])
axes[0].set_xticklabels(["Not Subscribed","Subscribed"])
style_ax(axes[0], "Call Duration by Outcome",
         xlabel="Outcome", ylabel="Duration (minutes)")

for label, color, sub in [("No", NEG_COLOR, df[df.y=="no"]["duration_min"]),
                           ("Yes", POS_COLOR, df[df.y=="yes"]["duration_min"])]:
    axes[1].hist(sub.clip(0,20), bins=40, density=True, alpha=0.55,
                 color=color, label=label, edgecolor="white", lw=0.3)
    sub.clip(0,20).plot.kde(ax=axes[1], color=color, lw=2)
axes[1].set_xlim(0, 20)
axes[1].legend(fontsize=10)
style_ax(axes[1], "Call Duration Density (0–20 min)",
         xlabel="Duration (minutes)", ylabel="Density")
plt.suptitle("Call Duration Analysis",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_06_duration_analysis.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 6 — Duration Analysis")

# ─── PLOT 7: Categorical Sub Rates (2×2) ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cat_features = [("education","Education Level"), ("marital","Marital Status"),
                ("housing","Has Housing Loan"), ("poutcome","Previous Campaign Outcome")]
for ax, (col, title) in zip(axes.flatten(), cat_features):
    grp = (df.groupby(col)["y_bin"]
             .agg(["mean","count"])
             .reset_index()
             .rename(columns={"mean":"rate","count":"n"})
             .sort_values("rate", ascending=False))
    bars = ax.bar(grp[col], grp["rate"]*100,
                  color=[POS_COLOR if v > grp["rate"].mean()
                         else PALETTE[1] for v in grp["rate"]],
                  edgecolor="white", width=0.5)
    ax.axhline(grp["rate"].mean()*100, color=ACCENT, ls="--", lw=1.6)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{row['rate']*100:.1f}%",
                ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")
    style_ax(ax, f"Subscription Rate by {title}", ylabel="Rate (%)", legend=False)
plt.suptitle("Subscription Rate Across Key Categorical Features",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_07_categorical_subrates.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 7 — Categorical Sub Rates")

# ─── PLOT 8: Correlation Heatmap ─────────────────────────────
num_cols = ["age","balance","duration","campaign","pdays","previous",
            "month_num","balance_log","contact_intensity","duration_min",
            "y_bin","was_contacted_before","high_balance"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(145, 10, as_cmap=True),
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"shrink":0.8}, annot_kws={"size":8})
style_ax(ax, "Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}plot_08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 8 — Correlation Heatmap")

# ─── PLOT 9: Age Group × Quarter Heatmap ─────────────────────
pivot_hm = (df.dropna(subset=["age_group","quarter"])
              .groupby(["age_group","quarter"])["y_bin"]
              .mean() * 100).unstack()
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_hm, annot=True, fmt=".1f",
            cmap=sns.light_palette(POS_COLOR, as_cmap=True),
            linewidths=0.5, ax=ax,
            cbar_kws={"label":"Subscription Rate (%)"})
style_ax(ax, "Subscription Rate (%) — Age Group × Quarter",
         xlabel="Quarter", ylabel="Age Group")
plt.tight_layout()
plt.savefig(f"{OUT}plot_09_agegroup_quarter_heatmap.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 9 — Age Group × Quarter Heatmap")

# ─── PLOT 10: Contact Strategy Analysis ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
contact_rate = (df.groupby("contact")["y_bin"]
                  .agg(["mean","count"]).reset_index()
                  .rename(columns={"mean":"rate","count":"n"}))
bars = axes[0].bar(contact_rate["contact"], contact_rate["rate"]*100,
                   color=PALETTE[:3], edgecolor="white", width=0.4)
for bar, (_, row) in zip(bars, contact_rate.iterrows()):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f"{row['rate']*100:.1f}%\n(n={int(row['n']):,})",
                 ha="center", fontsize=10, color=PRIMARY, fontweight="bold")
style_ax(axes[0], "Subscription Rate by Contact Type",
         xlabel="Contact Type", ylabel="Rate (%)")

sns.boxplot(data=df[df["campaign"]<=15], x="y", y="campaign",
            order=["no","yes"], palette=[NEG_COLOR, POS_COLOR],
            width=0.45, ax=axes[1])
axes[1].set_xticklabels(["Not Subscribed","Subscribed"])
style_ax(axes[1], "No. of Contacts This Campaign",
         xlabel="Outcome", ylabel="Number of Contacts")
plt.suptitle("Contact Strategy Analysis",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_10_contact_analysis.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 10 — Contact Analysis")


# ═══════════════════════════════════════════════════════════════
# 5. PREPROCESSING & TRAIN-TEST SPLIT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. PREPROCESSING & SPLITTING")
print("=" * 60)

cat_cols = ["job","marital","education","default","housing",
            "loan","contact","month","poutcome"]
df_model = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_model[col+"_enc"] = le.fit_transform(df_model[col].astype(str))

FEATURES = [
    "age", "balance_log", "duration_min", "campaign", "pdays",
    "previous", "month_num", "was_contacted_before", "high_balance",
    "contact_intensity", "has_loan_or_housing",
    "job_enc", "marital_enc", "education_enc",
    "housing_enc", "loan_enc", "contact_enc", "poutcome_enc"
]

X = df_model[FEATURES]
y = df_model["y_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
Xtr_s = sc.fit_transform(X_train)
Xte_s = sc.transform(X_test)

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Train — No: {(y_train==0).sum():,} | Yes: {(y_train==1).sum():,}")


# ═══════════════════════════════════════════════════════════════
# 6. MODELING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. MODELING")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest"      : RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = Xtr_s if name == "Logistic Regression" else X_train.values
    Xte = Xte_s if name == "Logistic Regression" else X_test.values
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)
    results[name] = {"model":model, "preds":preds,
                     "proba":proba, "auc":auc, "ap":ap}
    print(f"  {name:25s}  AUC={auc:.4f}  AP={ap:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n✔  Best model: {best_name}  (AUC={best['auc']:.4f})")


# ═══════════════════════════════════════════════════════════════
# 7. EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. EVALUATION")
print("=" * 60)

# ─── PLOT 11: Model Comparison ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names_list = list(results.keys())
aucs = [results[n]["auc"] for n in names_list]
aps  = [results[n]["ap"]  for n in names_list]

x = np.arange(len(names_list)); w = 0.35
bars1 = axes[0].bar(x - w/2, aucs, w, label="ROC-AUC",
                    color=PALETTE[1], edgecolor="white")
bars2 = axes[0].bar(x + w/2, aps,  w, label="Avg Precision",
                    color=PALETTE[2], edgecolor="white")
for bar, v in zip(list(bars1)+list(bars2), aucs+aps):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.3f}", ha="center", fontsize=10,
                 color=PRIMARY, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names_list, rotation=10)
axes[0].set_ylim(0.5, 1.0)
axes[0].legend(fontsize=10)
style_ax(axes[0], "Model Comparison — AUC & Avg Precision",
         ylabel="Score", legend=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models.items():
    Xtr = Xtr_s if name == "Logistic Regression" else X_train.values
    cv_scores[name] = cross_val_score(
        model, Xtr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

bp = axes[1].boxplot([cv_scores[n] for n in names_list],
                     labels=[n.replace(" ","\n") for n in names_list],
                     patch_artist=True, widths=0.4)
for patch, color in zip(bp["boxes"], PALETTE[:3]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color(PRIMARY); median.set_linewidth(2)
style_ax(axes[1], "5-Fold CV ROC-AUC Distribution", ylabel="ROC-AUC")
plt.suptitle("Model Performance Comparison",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_11_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 11 — Model Comparison")

# ─── PLOT 12: Confusion Matrix ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, best["preds"])
disp = ConfusionMatrixDisplay(cm, display_labels=["No (0)","Yes (1)"])
disp.plot(ax=ax, colorbar=False,
          cmap=sns.light_palette(POS_COLOR, as_cmap=True))
tn, fp, fn, tp = cm.ravel()
ax.text(1.15, 0.5,
        f"Precision : {tp/(tp+fp):.3f}\n"
        f"Recall    : {tp/(tp+fn):.3f}\n"
        f"F1-Score  : {2*tp/(2*tp+fp+fn):.3f}",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="center", color=PRIMARY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=ACCENT, alpha=0.3))
style_ax(ax, f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}plot_12_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 12 — Confusion Matrix")

# ─── PLOT 13: ROC + Precision-Recall Curves ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_roc = [PALETTE[0], PALETTE[2], PALETTE[4]]

for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC={res['auc']:.3f})")
axes[0].plot([0,1],[0,1], "k--", lw=1, label="Random")
axes[0].legend(fontsize=9, loc="lower right")
style_ax(axes[0], "ROC Curves — All Models",
         xlabel="False Positive Rate",
         ylabel="True Positive Rate", legend=False)

for (name, res), color in zip(results.items(), colors_roc):
    prec, rec, _ = precision_recall_curve(y_test, res["proba"])
    axes[1].plot(rec, prec, color=color, lw=2,
                 label=f"{name} (AP={res['ap']:.3f})")
axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1,
                label=f"Baseline ({y_test.mean():.3f})")
axes[1].legend(fontsize=9, loc="upper right")
style_ax(axes[1], "Precision-Recall Curves — All Models",
         xlabel="Recall", ylabel="Precision", legend=False)
plt.suptitle("Evaluation Curves",
             fontsize=15, fontweight="bold", color=PRIMARY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}plot_13_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 13 — ROC & PR Curves")

# ─── PLOT 14: Feature Importance ──────────────────────────────
fi = pd.Series(best["model"].feature_importances_,
               index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = [POS_COLOR if v >= fi.median() else PALETTE[1] for v in fi]
bars = ax.barh(fi.index, fi.values, color=colors_fi,
               edgecolor="white", height=0.65)
ax.axvline(fi.median(), color=ACCENT, ls="--", lw=1.8)
for bar, v in zip(bars, fi.values):
    ax.text(bar.get_width()+0.0003, bar.get_y()+bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8.5, color=PRIMARY)
patch_hi = mpatches.Patch(color=POS_COLOR, label="Above Median")
patch_lo = mpatches.Patch(color=PALETTE[1], label="Below Median")
ax.legend(handles=[patch_hi, patch_lo], fontsize=10)
style_ax(ax, f"Feature Importance — {best_name}", xlabel="Importance Score")
plt.tight_layout()
plt.savefig(f"{OUT}plot_14_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show(); print("✔  Plot 14 — Feature Importance")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE ✔")
print(f"  Best Model : {best_name}")
print(f"  AUC        : {best['auc']:.4f}")
print(f"  Avg Prec   : {best['ap']:.4f}")
print("=" * 60)
