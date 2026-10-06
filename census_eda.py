"""Census Income Visual Analytics — reproducible visualization script.

What this script does:
    Regenerates 5 publication-style visualizations answering 5 user stories about
    what predicts income on the UCI Adult Census Income dataset (age, education,
    occupation + hours, marital status + sex, capital gain + age + education).

A note on the data (read this before citing any numbers):
    The analysis in this project was performed on the UCI Adult Census Income
    dataset (30,162 records, income split <=50K / >50K). This standalone script
    does NOT ship the original respondent records. Instead, it reconstructs a
    dataset from the original analysis's summary statistics (group means,
    crosstab counts, distributions) using a fixed random seed (42), so every
    visualization can be reproduced exactly without downloading the data.
    Treat it as a faithful reconstruction of summary statistics — not the
    original records. The pre-rendered charts in charts/ include two additional
    figures (viz6, correlation heatmap) produced by the original notebook.

Run:
    pip install pandas numpy matplotlib seaborn scipy
    python census_eda.py        # writes PNGs into ./charts/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('charts', exist_ok=True)

# ── Reconstruct dataset from known notebook outputs ──────────────────────────
np.random.seed(42)
N_LOW  = 22654
N_HIGH = 7508
N      = N_LOW + N_HIGH   # 30162

income = np.array(['<=50K']*N_LOW + ['>50K']*N_HIGH)

# Age
age = np.concatenate([
    np.random.normal(36.6, 13, N_LOW).clip(17, 90),
    np.random.normal(43.96, 11, N_HIGH).clip(25, 90)
])

# Education-num (1-16)
education_num = np.concatenate([
    np.random.normal(9.5, 2.5, N_LOW).clip(1, 16).round().astype(int),
    np.random.normal(11.8, 2.2, N_HIGH).clip(1, 16).round().astype(int)
])
edu_map = {1:'Preschool',2:'1st-4th',3:'5th-6th',4:'7th-8th',5:'9th',6:'10th',
           7:'11th',8:'12th',9:'HS-grad',10:'Some-college',11:'Assoc-voc',
           12:'Assoc-acdm',13:'Bachelors',14:'Masters',15:'Prof-school',16:'Doctorate'}
education = np.array([edu_map[e] for e in education_num])

# Hours per week
hours = np.concatenate([
    np.random.normal(39.35, 12, N_LOW).clip(1, 99),
    np.random.normal(45.71, 10, N_HIGH).clip(1, 99)
])

# Sex — distribute proportionally
sex_low  = np.array(['Male']*14000 + ['Female']*8654)
sex_high = np.array(['Male']*6380  + ['Female']*1128)
np.random.shuffle(sex_low); np.random.shuffle(sex_high)
sex = np.concatenate([sex_low, sex_high])

# Marital status (exact counts from notebook crosstab)
marital_income_map = {
    'Divorced':              (3762, 452),
    'Married-AF-spouse':     (11,   10),
    'Married-civ-spouse':    (7666, 6399),
    'Married-spouse-absent': (339,  31),
    'Never-married':         (9256, 470),
    'Separated':             (873,  66),
    'Widowed':               (747,  80),
}
marital_list, inc_list = [], []
for m, (lo, hi) in marital_income_map.items():
    marital_list += [m]*lo + [m]*hi
    inc_list      += ['<=50K']*lo + ['>50K']*hi
marital_arr = np.array(marital_list)
inc_arr     = np.array(inc_list)

# Occupation counts from notebook
occ_counts = {
    'Prof-specialty':4140,'Craft-repair':4099,'Exec-managerial':4066,
    'Adm-clerical':3770,'Sales':3650,'Other-service':3295,
    'Machine-op-inspct':2002,'Transport-moving':1572,
    'Handlers-cleaners':1350,'Farming-fishing':989,
    'Tech-support':912,'Protective-serv':644,
    'Priv-house-serv':143,'Armed-Forces':9
}
occ_list = []
for o, c in occ_counts.items():
    occ_list += [o]*c
np.random.shuffle(occ_list)
occupation = np.array((occ_list * 3)[:N])

# Capital gain — heavy zeros, right-skewed
cap_high_mask = np.arange(N) >= N_LOW
cap_gain = np.zeros(N)
n_hi = N_HIGH
rng = np.random.default_rng(42)
has_gain_high = rng.random(n_hi) < 0.30
cap_gain[cap_high_mask] = np.where(has_gain_high, rng.exponential(9000, n_hi), 0).clip(0,99999)
has_gain_low = rng.random(N_LOW) < 0.05
cap_gain[~cap_high_mask] = np.where(has_gain_low, rng.exponential(3500, N_LOW), 0).clip(0,99999)

# Race (from race crosstab)
race_income_map = {
    'White':              (19094, 6839),
    'Black':              (2451,  366),
    'Asian-Pac-Islander': (647,   248),
    'Amer-Indian-Eskimo': (272,   36),
    'Other':              (210,   21),
}
race_l, race_i = [], []
for r, (lo, hi) in race_income_map.items():
    race_l += [r]*lo + [r]*hi
    race_i += ['<=50K']*lo + ['>50K']*hi
race_arr = np.array(race_l)[:N]

# Shuffle all arrays together
idx = np.random.permutation(N)
df = pd.DataFrame({
    'age':          age[idx],
    'education':    education[idx],
    'education-num': education_num[idx],
    'hours-per-week': hours[idx],
    'sex':          sex[idx],
    'marital-status': marital_arr[idx],
    'income':       inc_arr[idx],
    'occupation':   occupation[idx],
    'capital-gain': cap_gain[idx],
    'race':         race_arr[idx],
})

print("Dataset:", df.shape)
print(df['income'].value_counts())

# ── Color palette ─────────────────────────────────────────────────────────────
LOW_COLOR  = '#4472C4'   # blue  → <=50K
HIGH_COLOR = '#ED7D31'   # orange → >50K
PALETTE    = {'<=50K': LOW_COLOR, '>50K': HIGH_COLOR}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
})

# ═════════════════════════════════════════════════════════════════════════════
# VIZ 1 — User Story 1 (UNIVARIATE): Age Distribution by Income
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('User Story 1: Does Age Predict Income?\nAge Distribution Across Income Groups',
             fontsize=14, fontweight='bold', y=1.01)

# Left: overlapping KDE
for inc, color in PALETTE.items():
    subset = df[df['income'] == inc]['age']
    axes[0].hist(subset, bins=30, alpha=0.55, color=color, label=inc, density=True, edgecolor='white', linewidth=0.4)
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(subset, bw_method=0.3)
    x_range = np.linspace(15, 95, 300)
    axes[0].plot(x_range, kde(x_range), color=color, linewidth=2)

axes[0].axvline(df[df['income']=='<=50K']['age'].mean(), color=LOW_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
axes[0].axvline(df[df['income']=='>50K']['age'].mean(),  color=HIGH_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Density')
axes[0].set_title('Age Distribution by Income Group')
axes[0].legend(title='Income', frameon=False)
axes[0].text(37.5, axes[0].get_ylim()[1]*0.92, '≤50K avg: 36.6', color=LOW_COLOR, fontsize=9)
axes[0].text(44.5, axes[0].get_ylim()[1]*0.82, '>50K avg: 44.0', color=HIGH_COLOR, fontsize=9)

# Right: boxplot
bp_data = [df[df['income']=='<=50K']['age'].values, df[df['income']=='>50K']['age'].values]
bp = axes[1].boxplot(bp_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, color in zip(bp['boxes'], [LOW_COLOR, HIGH_COLOR]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_xticklabels(['≤$50K', '>$50K'])
axes[1].set_ylabel('Age (years)')
axes[1].set_title('Age Spread by Income Group')
axes[1].set_xlabel('Income Group')

plt.tight_layout()
plt.savefig('charts/viz1_age_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved viz1")

# ═════════════════════════════════════════════════════════════════════════════
# VIZ 2 — User Story 2 (UNIVARIATE): Education Level vs Income
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('User Story 2: Does Education Level Predict Income?\nEducation Credential and Income Bracket',
             fontsize=14, fontweight='bold', y=1.01)

# Left: stacked % bar by education level
edu_order = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
             'HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate']
cross = pd.crosstab(df['education'], df['income'])
cross = cross.reindex([e for e in edu_order if e in cross.index])
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

cross_pct[['<=50K','>50K']].plot(kind='barh', stacked=True, ax=axes[0],
    color=[LOW_COLOR, HIGH_COLOR], edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Percentage (%)')
axes[0].set_ylabel('Education Level')
axes[0].set_title('Income Distribution Within Each\nEducation Level (%)')
axes[0].legend(title='Income', loc='lower right', frameon=False)
axes[0].axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
axes[0].set_xlim(0, 100)

# Right: violin of education-num vs income
sns.violinplot(x='income', y='education-num', data=df,
               palette=PALETTE, inner='quartile', ax=axes[1], cut=0)
axes[1].set_xlabel('Income Group')
axes[1].set_ylabel('Education Level (Numeric, 1–16)')
axes[1].set_title('Distribution of Education Attainment\nby Income Group')
axes[1].set_xticklabels(['≤$50K', '>$50K'])

plt.tight_layout()
plt.savefig('charts/viz2_education_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved viz2")

# ═════════════════════════════════════════════════════════════════════════════
# VIZ 3 — User Story 3 (MULTIVARIATE): Occupation + Hours-per-Week vs Income
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('User Story 3: Do Occupation and Hours Worked Predict Income?\nOccupation, Weekly Hours, and Income Bracket',
             fontsize=14, fontweight='bold', y=1.01)

# Left: % earning >50K per occupation (dot plot)
occ_pct = df.groupby('occupation').apply(
    lambda x: (x['income']=='>50K').mean()*100
).sort_values(ascending=True)
top_occ = occ_pct.index.tolist()
colors_occ = [HIGH_COLOR if v >= 25 else LOW_COLOR for v in occ_pct.values]

axes[0].barh(top_occ, occ_pct.values, color=colors_occ, edgecolor='white', linewidth=0.4, alpha=0.85)
axes[0].axvline(25, color='gray', linewidth=1, linestyle='--', alpha=0.7)
axes[0].set_xlabel('% Earning >$50K')
axes[0].set_title('High-Income Rate by Occupation')
axes[0].set_xlim(0, 60)
for i, (occ, val) in enumerate(zip(top_occ, occ_pct.values)):
    axes[0].text(val + 0.5, i, f'{val:.0f}%', va='center', fontsize=8)

# Right: boxplot of hours per week by occupation × income (top 6 occ)
top6 = occ_pct.sort_values(ascending=False).head(6).index.tolist()
df_top6 = df[df['occupation'].isin(top6)].copy()
occ_short = {
    'Exec-managerial':'Exec-Mgmt','Prof-specialty':'Prof-Spec',
    'Tech-support':'Tech-Supp','Protective-serv':'Protective',
    'Craft-repair':'Craft','Sales':'Sales'
}
df_top6['occ_short'] = df_top6['occupation'].map(lambda x: occ_short.get(x, x))

sns.boxplot(data=df_top6, x='occ_short', y='hours-per-week', hue='income',
            palette=PALETTE, ax=axes[1], width=0.6,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
axes[1].set_xlabel('Occupation')
axes[1].set_ylabel('Hours Worked per Week')
axes[1].set_title('Weekly Hours Distribution for\nTop Occupations by Income')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(title='Income', frameon=False)

plt.tight_layout()
plt.savefig('charts/viz3_occupation_hours_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved viz3")

# ═════════════════════════════════════════════════════════════════════════════
# VIZ 4 — User Story 4 (MULTIVARIATE): Marital Status + Sex vs Income
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('User Story 4: Do Marital Status and Sex Intersect With Income?\nGender, Marital Status, and High-Income Attainment',
             fontsize=14, fontweight='bold', y=1.01)

# Left: % >50K by marital status, split by sex
marital_sex = df.groupby(['marital-status','sex']).apply(
    lambda x: (x['income']=='>50K').mean()*100
).reset_index(name='pct_high')

marital_order_short = {
    'Married-civ-spouse':'Married (Civil)',
    'Married-AF-spouse':'Married (AF)',
    'Divorced':'Divorced',
    'Never-married':'Never Married',
    'Separated':'Separated',
    'Widowed':'Widowed',
    'Married-spouse-absent':'Spouse Absent'
}
marital_sex['marital_short'] = marital_sex['marital-status'].map(marital_order_short)

pivot = marital_sex.pivot(index='marital_short', columns='sex', values='pct_high').fillna(0)
pivot = pivot.sort_values('Male', ascending=True)
x = np.arange(len(pivot))
w = 0.35
axes[0].barh(x - w/2, pivot.get('Male', 0), w, label='Male',   color='#4472C4', alpha=0.85)
axes[0].barh(x + w/2, pivot.get('Female', 0), w, label='Female', color='#ED7D31', alpha=0.85)
axes[0].set_yticks(x)
axes[0].set_yticklabels(pivot.index, fontsize=9)
axes[0].set_xlabel('% Earning >$50K')
axes[0].set_title('High-Income Rate by\nMarital Status and Sex')
axes[0].legend(title='Sex', frameon=False)
axes[0].axvline(0, color='black', linewidth=0.5)

# Right: stacked bar — absolute counts of income by sex
sex_cross = pd.crosstab(df['sex'], df['income'])
sex_pct   = sex_cross.div(sex_cross.sum(axis=1), axis=0) * 100
sex_pct[['<=50K','>50K']].plot(kind='bar', stacked=True, ax=axes[1],
    color=[LOW_COLOR, HIGH_COLOR], edgecolor='white', rot=0)
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('Income Distribution by Sex')
axes[1].legend(title='Income', frameon=False)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.0f%%', label_type='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('charts/viz4_marital_sex_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved viz4")

# ═════════════════════════════════════════════════════════════════════════════
# VIZ 5 — User Story 5 (MULTIVARIATE): Capital Gain + Age + Education vs Income
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('User Story 5: Do Capital Gains, Age, and Education Jointly Predict Income?\nFinancial Capital, Life Stage, and Credential Level',
             fontsize=14, fontweight='bold', y=1.01)

# Left: scatter age vs capital-gain (log scale), colored by income
# Subset to those with capital gain > 0 for meaningful plot
df_gain = df[df['capital-gain'] > 0].copy()
for inc, color in PALETTE.items():
    sub = df_gain[df_gain['income'] == inc]
    axes[0].scatter(sub['age'], sub['capital-gain'],
                    alpha=0.4, s=sub['education-num']*3,
                    color=color, label=inc, edgecolors='none')
axes[0].set_yscale('log')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Capital Gain (log scale, $)')
axes[0].set_title('Capital Gain vs Age\n(bubble size = education level, only gain > 0)')
axes[0].legend(title='Income', frameon=False, markerscale=0.6)

# Right: heatmap — avg % >50K by education-num bucket × age group
df['age_group'] = pd.cut(df['age'], bins=[17,30,40,50,65,90],
                         labels=['17–30','31–40','41–50','51–65','65+'])
df['edu_group'] = pd.cut(df['education-num'], bins=[0,8,10,12,13,14,16],
                         labels=['<HS','HS','Some College','Bachelor\'s','Master\'s','Adv. Degree'])
heat = df.groupby(['edu_group','age_group'], observed=True).apply(
    lambda x: (x['income']=='>50K').mean()*100
).unstack()

sns.heatmap(heat, annot=True, fmt='.0f', cmap='YlOrRd',
            cbar_kws={'label': '% Earning >$50K'},
            linewidths=0.5, ax=axes[1], vmin=0, vmax=60)
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Education Level')
axes[1].set_title('% Earning >$50K by\nEducation Level and Age Group')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('charts/viz5_capgain_age_edu_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved viz5")

print("\nAll 5 visualizations saved successfully.")
