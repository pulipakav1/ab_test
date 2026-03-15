import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/ab_data.csv')

print("Data Overview")

print(f"Total rows        : {len(df):,}")
print(f"Unique users      : {df['user_id'].nunique():,}")
print(f"Duplicate users   : {len(df) - df['user_id'].nunique():,}")

mismatched = df[
    ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page')) |
    ((df['group'] == 'control')   & (df['landing_page'] == 'new_page'))
]
print(f"Mismatched rows   : {len(mismatched):,}")

df = df.drop_duplicates(subset='user_id', keep='first')
df = df[~df.index.isin(mismatched.index)]
print(f"Clean rows        : {len(df):,}")

print("Group balance check")

groups = df.groupby('group').agg(
    users=('user_id', 'count'),
    conversions=('converted', 'sum'),
    conversion_rate=('converted', 'mean')
).round(4)
print(groups)

control   = df[df['group'] == 'control']['converted']
treatment = df[df['group'] == 'treatment']['converted']

print("Hypothesis testing")

p_control   = control.mean()
p_treatment = treatment.mean()
n_control   = len(control)
n_treatment = len(treatment)

p_pool = (control.sum() + treatment.sum()) / (n_control + n_treatment)
se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
z_stat = (p_treatment - p_control) / se
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

ci_low  = (p_treatment - p_control) - 1.96 * se
ci_high = (p_treatment - p_control) + 1.96 * se

print(f"Control rate      : {p_control:.4f}  ({p_control*100:.2f}%)")
print(f"Treatment rate    : {p_treatment:.4f}  ({p_treatment*100:.2f}%)")
print(f"Absolute lift     : {(p_treatment - p_control)*100:+.4f}%")
print(f"Relative lift     : {((p_treatment - p_control)/p_control)*100:+.2f}%")
print(f"Z-statistic       : {z_stat:.4f}")
print(f"P-value           : {p_value:.4f}")
print(f"95% CI            : [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")
print(f"Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

print("Effect size (Cohen's h)")

cohens_h = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))
print(f"Cohen's h         : {cohens_h:.4f}")
print(f"Interpretation    : {'Small' if abs(cohens_h) < 0.2 else 'Medium' if abs(cohens_h) < 0.5 else 'Large'} effect")

print("Power Analysis")

alpha     = 0.05
power     = 0.80
z_alpha   = norm.ppf(1 - alpha/2)
z_beta    = norm.ppf(power)
mde       = 0.01
p_base    = p_control
required_n = (2 * p_base * (1 - p_base) * (z_alpha + z_beta)**2) / mde**2

print(f"Minimum detectable effect : {mde*100}%")
print(f"Required sample per group : {int(required_n):,}")
print(f"Actual sample per group   : {min(n_control, n_treatment):,}")
print(f"Adequately powered        : {'YES' if min(n_control, n_treatment) >= required_n else 'NO'}")

print("Business impact")
daily_visitors   = 10000
revenue_per_conv = 25
lift             = p_treatment - p_control
daily_impact     = daily_visitors * lift * revenue_per_conv
annual_impact    = daily_impact * 365

print(f"Assumed daily visitors    : {daily_visitors:,}")
print(f"Assumed revenue/conversion: ${revenue_per_conv}")
print(f"Daily revenue impact      : ${daily_impact:,.0f}")
print(f"Annual revenue impact     : ${annual_impact:,.0f}")
print(f"Recommendation            : {'DO NOT LAUNCH — no significant improvement detected' if p_value >= 0.05 else 'LAUNCH — significant improvement detected'}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('E-commerce A/B Test Analysis — New Page vs Old Page', fontsize=13, fontweight='bold')

ax1 = axes[0]
rates = [p_control * 100, p_treatment * 100]
bars = ax1.bar(['Control\n(Old Page)', 'Treatment\n(New Page)'], rates, color=['#d0d0d0', '#4472C4'], width=0.5)
ax1.set_ylabel('Conversion Rate (%)')
ax1.set_title('Conversion Rate by Group')
ax1.set_ylim(0, max(rates) * 1.3)
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{rate:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
df['date'] = pd.to_datetime(df['timestamp']).dt.date
daily = df.groupby(['date', 'group'])['converted'].mean().unstack()
daily['control'].plot(ax=ax2, label='Control', color='#d0d0d0', linewidth=2)
daily['treatment'].plot(ax=ax2, label='Treatment', color='#4472C4', linewidth=2)
ax2.set_title('Daily Conversion Rate Over Time')
ax2.set_ylabel('Conversion Rate')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax3 = axes[2]
lift_val = (p_treatment - p_control) * 100
ci_vals  = [abs(ci_low * 100), abs(ci_high * 100)]
ax3.barh(['Conversion Lift'], [lift_val], xerr=[[abs(ci_low*100)], [abs(ci_high*100)]], color='#4472C4' if lift_val > 0 else '#E05C5C', height=0.4, capsize=8)
ax3.axvline(0, color='black', linewidth=1)
ax3.set_title('Lift with 95% Confidence Interval')
ax3.set_xlabel('Absolute Lift (%)')
ax3.grid(axis='x', alpha=0.3)
significance = f"p = {p_value:.4f} — {'Significant' if p_value < 0.05 else 'Not Significant'}"
ax3.text(0.5, -0.25, significance, transform=ax3.transAxes, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('reports/ab_test_results.png', dpi=150, bbox_inches='tight')
print("Saved reports/ab_test_results.png")