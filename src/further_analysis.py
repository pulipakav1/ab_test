import pandas as pd
import numpy as np
from scipy.stats import norm, beta as beta_dist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/ab_data.csv')
df = df.drop_duplicates(subset='user_id', keep='first')
mismatched = df[
    ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page')) |
    ((df['group'] == 'control')   & (df['landing_page'] == 'new_page'))
]
df = df[~df.index.isin(mismatched.index)]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df = df.sort_values('timestamp')

control   = df[df['group'] == 'control']['converted']
treatment = df[df['group'] == 'treatment']['converted']

dates = sorted(df['date'].unique())
cumulative = []
for i, cutoff in enumerate(dates):
    subset = df[df['date'] <= cutoff]
    c = subset[subset['group'] == 'control']['converted']
    t = subset[subset['group'] == 'treatment']['converted']
    if len(c) < 100 or len(t) < 100:
        continue
    p_pool = (c.sum() + t.sum()) / (len(c) + len(t))
    se = np.sqrt(p_pool * (1 - p_pool) * (1/len(c) + 1/len(t)))
    z  = (t.mean() - c.mean()) / se
    cumulative.append({
        'date'        : cutoff,
        'day'         : i + 1,
        'lift'        : t.mean() - c.mean(),
        'p_value'     : 2 * (1 - norm.cdf(abs(z)))
    })

novelty_df = pd.DataFrame(cumulative)
early = novelty_df[novelty_df['day'] <= 7]['lift'].mean()
late  = novelty_df[novelty_df['day'] >  7]['lift'].mean()

alpha_c = 1 + control.sum()
beta_c  = 1 + len(control) - control.sum()
alpha_t = 1 + treatment.sum()
beta_t  = 1 + len(treatment) - treatment.sum()

n_samples  = 100000
samples_c  = beta_dist.rvs(alpha_c, beta_c, size=n_samples)
samples_t  = beta_dist.rvs(alpha_t, beta_t, size=n_samples)
prob_better = (samples_t > samples_c).mean()
exp_lift    = (samples_t - samples_c).mean()
ci_low      = np.percentile(samples_t - samples_c, 2.5)
ci_high     = np.percentile(samples_t - samples_c, 97.5)

p_c    = control.mean()
p_t    = treatment.mean()
p_pool = (control.sum() + treatment.sum()) / (len(control) + len(treatment))
se     = np.sqrt(p_pool * (1 - p_pool) * (1/len(control) + 1/len(treatment)))
z_stat = (p_t - p_c) / se
p_val  = 2 * (1 - norm.cdf(abs(z_stat)))

print(f"control conversion rate   : {p_c:.4f}")
print(f"treatment conversion rate : {p_t:.4f}")
print(f"absolute lift             : {(p_t - p_c)*100:+.4f}%")
print(f"p-value                   : {p_val:.4f}")
print(f"significant               : {'yes' if p_val < 0.05 else 'no'}")
print()
print(f"novelty effect — first 7 days avg lift : {early*100:+.4f}%")
print(f"novelty effect — after 7 days avg lift : {late*100:+.4f}%")
print(f"novelty effect detected                : {'yes' if early > late else 'no'}")
print()
print(f"p(treatment better) — bayesian : {prob_better*100:.1f}%")
print(f"expected lift — bayesian       : {exp_lift*100:+.4f}%")
print(f"95% credible interval          : [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")
print(f"bayesian recommendation        : {'launch' if prob_better > 0.95 else 'do not launch'}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('A/B Test — Advanced Analysis', fontsize=13, fontweight='bold')

ax1 = axes[0]
ax1.plot(novelty_df['day'], novelty_df['lift'] * 100,
         color='#4472C4', linewidth=2, marker='o', markersize=3)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.axvline(7, color='#E05C5C', linewidth=1, linestyle='--', label='Day 7')
ax1.fill_between(novelty_df['day'], novelty_df['lift'] * 100, 0,
                 alpha=0.1, color='#4472C4')
ax1.set_xlabel('Days running')
ax1.set_ylabel('Cumulative lift (%)')
ax1.set_title('Novelty effect check')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
x = np.linspace(0.10, 0.14, 1000)
ax2.plot(x, beta_dist.pdf(x, alpha_c, beta_c),
         color='#d0d0d0', linewidth=2, label=f'Control ({p_c:.4f})')
ax2.plot(x, beta_dist.pdf(x, alpha_t, beta_t),
         color='#4472C4', linewidth=2, label=f'Treatment ({p_t:.4f})')
ax2.fill_between(x, beta_dist.pdf(x, alpha_c, beta_c), alpha=0.2, color='#d0d0d0')
ax2.fill_between(x, beta_dist.pdf(x, alpha_t, beta_t), alpha=0.2, color='#4472C4')
ax2.set_xlabel('Conversion rate')
ax2.set_ylabel('Density')
ax2.set_title('Posterior distributions')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = axes[2]
lift_samples = (samples_t - samples_c) * 100
ax3.hist(lift_samples, bins=80, color='#4472C4', alpha=0.7, edgecolor='none')
ax3.axvline(0, color='#E05C5C', linewidth=2, linestyle='--', label='No lift')
ax3.axvline(exp_lift * 100, color='#4472C4', linewidth=2,
            label=f'Expected: {exp_lift*100:+.3f}%')
ax3.axvline(ci_low * 100,  color='gray', linewidth=1, linestyle=':')
ax3.axvline(ci_high * 100, color='gray', linewidth=1, linestyle=':', label='95% CI')
ax3.set_xlabel('Lift (%)')
ax3.set_ylabel('Frequency')
ax3.set_title(f'Posterior lift — P(better) = {prob_better*100:.1f}%')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/advanced_analysis.png', dpi=150, bbox_inches='tight')
print("\nplot saved to reports/advanced_analysis.png")