"""
Data Cleaning & Visualization Project
======================================
Author  : [Your Name]
Due Date: 28 Apr 2026
Dataset : Synthetic Retail Sales Dataset (2000 records)

Libraries: pandas, numpy, matplotlib, seaborn
Run     : python data_cleaning_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                      "axes.spines.right": False})

# ─────────────────────────────────────────────
# 1. GENERATE RAW (DIRTY) DATASET
# ─────────────────────────────────────────────
N = 2000

categories   = ["Electronics", "Apparel", "Grocery", "Home & Garden",
                "Sports", "Books", "Toys"]
regions      = ["North", "South", "East", "West"]

raw = pd.DataFrame({
    "order_id"     : range(1, N + 1),
    "order_date"   : pd.date_range("2025-01-01", periods=N, freq="4h"),
    "customer_age" : np.random.randint(18, 70, N).astype(float),
    "category"     : np.random.choice(categories, N),
    "region"       : np.random.choice(regions, N),
    "revenue"      : np.abs(np.random.normal(2300, 800, N)),
    "discount_pct" : np.random.uniform(0, 40, N),
    "units_sold"   : np.random.randint(1, 20, N),
})

# ── Inject dirt ──────────────────────────────
# Missing values
missing_idx = np.random.choice(N, 182, replace=False)
raw.loc[missing_idx, "customer_age"] = np.nan

cat_missing = np.random.choice(N, 67, replace=False)
raw.loc[cat_missing, "category"] = np.nan

# Outliers in revenue (extreme high values)
outlier_idx = np.random.choice(N, 47, replace=False)
raw.loc[outlier_idx, "revenue"] = np.random.uniform(15000, 50000, 47)

# Outliers in discount_pct (>75%)
disc_outlier = np.random.choice(N, 17, replace=False)
raw.loc[disc_outlier, "discount_pct"] = np.random.uniform(80, 120, 17)

# Duplicate rows
dup_rows = raw.sample(89, random_state=7)
raw = pd.concat([raw, dup_rows], ignore_index=True)

# Corrupt date strings
date_corrupt = np.random.choice(len(raw), 23, replace=False)
raw.loc[date_corrupt, "order_date"] = "INVALID_DATE"

print("=" * 55)
print("  STEP 1 — Raw Dataset Inspection")
print("=" * 55)
print(f"  Shape          : {raw.shape}")
print(f"  Columns        : {list(raw.columns)}")
print(f"  Missing values :\n{raw.isnull().sum()}")
print(f"  Duplicates     : {raw.duplicated().sum()}")
print()

# ─────────────────────────────────────────────
# 2. CLEANING — Step by step
# ─────────────────────────────────────────────

df = raw.copy()

# ── 2a. Fix date column ──────────────────────
print("=" * 55)
print("  STEP 2 — Handling Dates")
print("=" * 55)
before = df.shape[0]
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
bad_dates = df["order_date"].isna().sum()
df.dropna(subset=["order_date"], inplace=True)
print(f"  Rows with invalid dates dropped : {bad_dates}")

# ── 2b. Drop duplicates ──────────────────────
print()
print("=" * 55)
print("  STEP 3 — Removing Duplicates")
print("=" * 55)
before_dup = df.shape[0]
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Duplicates removed : {before_dup - df.shape[0]}")

# ── 2c. Impute missing values ────────────────
print()
print("=" * 55)
print("  STEP 4 — Imputing Missing Values")
print("=" * 55)
age_median = df["customer_age"].median()
df["customer_age"].fillna(age_median, inplace=True)
print(f"  customer_age  : filled {df['customer_age'].isna().sum()} NaN → median ({age_median:.1f})")

for region in df["region"].unique():
    mode_cat = df.loc[df["region"] == region, "category"].mode()
    if len(mode_cat):
        df.loc[(df["region"] == region) & df["category"].isna(), "category"] = mode_cat[0]
df["category"].fillna(df["category"].mode()[0], inplace=True)
print(f"  category      : filled using region-level mode")

# ── 2d. Handle outliers (IQR) ────────────────
print()
print("=" * 55)
print("  STEP 5 — Outlier Removal (IQR Method)")
print("=" * 55)

def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    before = len(data)
    data = data[(data[col] >= lower) & (data[col] <= upper)]
    print(f"  {col:20s}: removed {before - len(data)} outliers  "
          f"[{lower:.1f}, {upper:.1f}]")
    return data

df = remove_outliers_iqr(df, "revenue")

# Cap discount_pct instead of dropping
df["discount_pct"] = df["discount_pct"].clip(upper=75)
print(f"  {'discount_pct':20s}: capped at 75%")

print()
print("=" * 55)
print("  CLEANING COMPLETE")
print("=" * 55)
print(f"  Raw records   : {raw.shape[0]}")
print(f"  Clean records : {df.shape[0]}")
print(f"  Removed       : {raw.shape[0] - df.shape[0]} "
      f"({(raw.shape[0]-df.shape[0])/raw.shape[0]*100:.1f}%)")
print(f"  Remaining NaN : {df.isnull().sum().sum()}")
print()

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["month"]         = df["order_date"].dt.month
df["month_name"]    = df["order_date"].dt.strftime("%b")
df["quarter"]       = df["order_date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
df["revenue_per_unit"] = (df["revenue"] / df["units_sold"]).round(2)
df["age_group"] = pd.cut(df["customer_age"],
                          bins=[17,24,31,38,45,52,59,100],
                          labels=["18-24","25-31","32-38","39-45","46-52","53-59","60+"])

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
PALETTE = ["#378ADD","#1D9E75","#EF9F27","#D85A30","#7F77DD","#D4537E","#639922"]

# ── Figure 1: Dashboard (2×3) ────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Retail Sales — Data Cleaning & Visualization Report",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 1. Revenue by category
ax1 = fig.add_subplot(gs[0, 0])
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
bars = ax1.bar(range(len(cat_rev)), cat_rev.values / 1e5, color=PALETTE[:len(cat_rev)],
               edgecolor="none", width=0.6)
ax1.set_xticks(range(len(cat_rev)))
ax1.set_xticklabels(cat_rev.index, rotation=35, ha="right", fontsize=8)
ax1.set_ylabel("Revenue (₹ Lakhs)")
ax1.set_title("Revenue by Category", fontsize=11, pad=8)
for bar, val in zip(bars, cat_rev.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"₹{val/1e5:.1f}L", ha="center", fontsize=7, color="#5F5E5A")

# 2. Monthly trend
ax2 = fig.add_subplot(gs[0, 1])
monthly = df.groupby("month")["revenue"].sum() / 1e5
ax2.plot(monthly.index, monthly.values, color="#378ADD", linewidth=2,
         marker="o", markersize=5, markerfacecolor="white", markeredgewidth=2)
ax2.fill_between(monthly.index, monthly.values, alpha=0.12, color="#378ADD")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=9)
ax2.set_ylabel("Revenue (₹ Lakhs)")
ax2.set_title("Monthly Revenue Trend", fontsize=11, pad=8)

# 3. Region breakdown
ax3 = fig.add_subplot(gs[0, 2])
reg_rev = df.groupby("region")["revenue"].sum()
wedges, texts, autotexts = ax3.pie(reg_rev.values, labels=reg_rev.index,
    autopct="%1.1f%%", colors=PALETTE[:4], startangle=90,
    wedgeprops={"linewidth":1.5,"edgecolor":"white"})
for t in autotexts:
    t.set_fontsize(8)
ax3.set_title("Revenue by Region", fontsize=11, pad=8)

# 4. Age group histogram
ax4 = fig.add_subplot(gs[1, 0])
age_counts = df["age_group"].value_counts().sort_index()
ax4.bar(age_counts.index, age_counts.values, color="#1D9E75", edgecolor="none", width=0.6)
ax4.set_xlabel("Age Group")
ax4.set_ylabel("Customers")
ax4.set_title("Customer Age Distribution", fontsize=11, pad=8)

# 5. Revenue boxplot by category
ax5 = fig.add_subplot(gs[1, 1])
data_by_cat = [df[df["category"]==c]["revenue"].values for c in cat_rev.index]
bp = ax5.boxplot(data_by_cat, patch_artist=True, medianprops={"color":"white","linewidth":2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax5.set_xticklabels([c[:6] for c in cat_rev.index], rotation=35, ha="right", fontsize=8)
ax5.set_ylabel("Revenue (₹)")
ax5.set_title("Revenue Spread by Category", fontsize=11, pad=8)

# 6. Heatmap — avg revenue by region × category
ax6 = fig.add_subplot(gs[1, 2])
pivot = df.pivot_table(values="revenue", index="region",
                        columns="category", aggfunc="mean")
sns.heatmap(pivot, ax=ax6, cmap="Blues", annot=True, fmt=".0f",
            linewidths=0.5, cbar_kws={"shrink": 0.75}, annot_kws={"size":7})
ax6.set_title("Avg Revenue: Region × Category", fontsize=11, pad=8)
ax6.set_xlabel("")
ax6.set_ylabel("")
ax6.tick_params(axis="x", rotation=35, labelsize=7)
ax6.tick_params(axis="y", rotation=0, labelsize=8)

plt.savefig("sales_dashboard.png", bbox_inches="tight", facecolor="white")
print("  [Saved] sales_dashboard.png")

# ── Figure 2: Outlier Analysis ───────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.suptitle("Outlier Detection — Revenue Column", fontsize=13, fontweight="bold")

# Before
axes[0].hist(raw["revenue"].dropna(), bins=60, color="#E24B4A", alpha=0.75, edgecolor="none")
axes[0].set_title("Before Cleaning", fontsize=11)
axes[0].set_xlabel("Revenue (₹)")
axes[0].set_ylabel("Frequency")

# After
axes[1].hist(df["revenue"], bins=60, color="#1D9E75", alpha=0.75, edgecolor="none")
axes[1].set_title("After Cleaning (IQR Filter)", fontsize=11)
axes[1].set_xlabel("Revenue (₹)")

plt.tight_layout()
plt.savefig("outlier_analysis.png", bbox_inches="tight", facecolor="white")
print("  [Saved] outlier_analysis.png")

# ── Figure 3: Missing Value Heatmap ──────────
fig3, ax = plt.subplots(figsize=(10, 3))
missing_df = raw.isnull().astype(int).sample(200, random_state=1)
sns.heatmap(missing_df.T, cbar=False, ax=ax, cmap=["#E6F1FB","#E24B4A"],
            linewidths=0, xticklabels=False)
ax.set_title("Missing Value Pattern — Raw Dataset (sample of 200 rows)", fontsize=11)
ax.set_ylabel("Column")
plt.tight_layout()
plt.savefig("missing_heatmap.png", bbox_inches="tight", facecolor="white")
print("  [Saved] missing_heatmap.png")

# ─────────────────────────────────────────────
# 5. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  DESCRIPTIVE STATISTICS (Clean Dataset)")
print("=" * 55)
print(df[["revenue","customer_age","discount_pct","units_sold"]].describe().round(2))
print()
print("  Top 3 revenue categories:")
print(cat_rev.head(3).to_string())
print()
print("  Correlation matrix (numeric columns):")
print(df[["revenue","customer_age","discount_pct","units_sold"]].corr().round(3))

# Save cleaned dataset
df.to_csv("cleaned_sales_data.csv", index=False)
print()
print("  [Saved] cleaned_sales_data.csv")
print()
print("  All done! Open the PNG files to view your visualizations.")
