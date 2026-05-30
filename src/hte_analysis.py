import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def load_clean_data(path='data/ab_data.csv'):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset='user_id', keep='first')
    mismatched = df[
        ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page')) |
        ((df['group'] == 'control')   & (df['landing_page'] == 'new_page'))
    ]
    df = df[~df.index.isin(mismatched.index)]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )
    first_seen = df.groupby('user_id')['date'].min().reset_index()
    first_seen.columns = ['user_id', 'first_seen_date']
    df = df.merge(first_seen, on='user_id')
    df['is_new_user'] = df['date'] == df['first_seen_date']
    return df


def z_test(control_series, treatment_series):
    p_c = control_series.mean()
    p_t = treatment_series.mean()
    n_c = len(control_series)
    n_t = len(treatment_series)
    p_pool = (control_series.sum() + treatment_series.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se == 0:
        return p_c, p_t, 0.0, 1.0, 0.0, 0.0
    z = (p_t - p_c) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    ci_low  = (p_t - p_c) - 1.96 * se
    ci_high = (p_t - p_c) + 1.96 * se
    return p_c, p_t, z, p_val, ci_low, ci_high


def segment_test(df, segment_col):
    results = []
    for val in sorted(df[segment_col].dropna().unique()):
        subset = df[df[segment_col] == val]
        c = subset[subset['group'] == 'control']['converted']
        t = subset[subset['group'] == 'treatment']['converted']
        if len(c) < 30 or len(t) < 30:
            continue
        p_c, p_t, z, p_val, ci_low, ci_high = z_test(c, t)
        results.append({
            'segment': str(val),
            'n_control': len(c),
            'n_treatment': len(t),
            'rate_control': round(p_c * 100, 3),
            'rate_treatment': round(p_t * 100, 3),
            'lift_pct': round((p_t - p_c) * 100, 4),
            'ci_low': round(ci_low * 100, 4),
            'ci_high': round(ci_high * 100, 4),
            'p_value': round(p_val, 4),
            'significant': p_val < 0.05
        })
    return pd.DataFrame(results)


def plot_hte(results_dict, output_path='reports/hte_analysis.png'):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle('Heterogeneous Treatment Effects — Segment-Level Analysis', fontsize=13, fontweight='bold')
    if n == 1:
        axes = [axes]

    for ax, (title, df) in zip(axes, results_dict.items()):
        colors = ['#2ecc71' if s else '#e74c3c' for s in df['significant']]
        bars = ax.barh(df['segment'], df['lift_pct'], color=colors, height=0.5)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        for bar, row in zip(bars, df.itertuples()):
            ax.errorbar(
                row.lift_pct, bar.get_y() + bar.get_height() / 2,
                xerr=[[row.lift_pct - row.ci_low], [row.ci_high - row.lift_pct]],
                fmt='none', color='black', capsize=4, linewidth=1.2
            )
        ax.set_title(title)
        ax.set_xlabel('Lift (%)')
        ax.grid(axis='x', alpha=0.3)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#2ecc71', label='Significant (p<0.05)'),
            Patch(color='#e74c3c', label='Not significant')
        ], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")


def main():
    df = load_clean_data()

    segments = {
        'User Type (New vs Returning)': 'is_new_user',
        'Time of Day': 'time_of_day',
        'Day Type (Weekday vs Weekend)': 'is_weekend',
    }

    all_results = {}
    for label, col in segments.items():
        results = segment_test(df, col)
        if not results.empty:
            all_results[label] = results
            print(f"\n--- {label} ---")
            print(results.to_string(index=False))

    if all_results:
        plot_hte(all_results)

    # Summary: flag any segment with significant positive lift
    print("\n--- Key Finding ---")
    for label, res in all_results.items():
        winners = res[(res['significant']) & (res['lift_pct'] > 0)]
        if not winners.empty:
            for _, row in winners.iterrows():
                print(f"  {label} | {row['segment']}: +{row['lift_pct']:.3f}% lift (p={row['p_value']:.4f})")
        else:
            print(f"  {label}: no segment shows significant positive lift")


if __name__ == '__main__':
    main()
