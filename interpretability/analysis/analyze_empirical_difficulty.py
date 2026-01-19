#!/usr/bin/env python3
"""
Analyze what determines empirical difficulty.

Key questions:
1. What spatial features correlate with empirical difficulty (completion rate)?
2. What features correlate with the agent's actual choice?
3. Why do empirical and geometric difficulty sometimes disagree?
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def load_results(filepath):
    """Load empirical difficulty results."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_features(results, test_type='optvar'):
    """Extract features from each configuration."""
    data = results[test_type]['results']
    rows = []

    for config in data:
        difficulty = config['difficulty']
        episode = config['episode']

        # Find the two intermediate zone names
        int_zones = [k for k in difficulty.keys()
                     if k not in ['empirically_easier', 'difficulty_diff', 'steps_diff', 'geometrically_easier']]

        if len(int_zones) < 2:
            continue

        z0, z1 = int_zones[0], int_zones[1]
        d0, d1 = difficulty[z0], difficulty[z1]

        # Extract positions
        pos0 = np.array(d0['position'])
        pos1 = np.array(d1['position'])

        # For each zone, compute features
        for zone_name, zone_data, other_data in [(z0, d0, d1), (z1, d1, d0)]:
            pos = np.array(zone_data['position'])

            row = {
                'config_id': config['config_id'],
                'zone_name': zone_name,
                'int_color': config['int_color'],
                'goal_color': config['goal_color'],

                # Raw features
                'd_agent': zone_data['d_agent'],
                'd_goal': zone_data['d_goal'],
                'geometric_total': zone_data['geometric_total'],
                'pos_x': pos[0],
                'pos_y': pos[1],

                # Empirical difficulty
                'completion_rate': zone_data['completion_rate'],
                'mean_steps': zone_data['mean_steps'],

                # Relative features (compared to other intermediate)
                'd_agent_diff': zone_data['d_agent'] - other_data['d_agent'],
                'd_goal_diff': zone_data['d_goal'] - other_data['d_goal'],
                'geometric_diff': zone_data['geometric_total'] - other_data['geometric_total'],

                # Labels
                'is_empirically_easier': zone_name == difficulty.get('empirically_easier'),
                'is_geometrically_easier': zone_name == difficulty.get('geometrically_easier'),
                'was_chosen': zone_name == episode['chosen_int'],
            }

            # Distance from origin (agent start is near origin)
            row['dist_from_origin'] = np.sqrt(pos[0]**2 + pos[1]**2)

            # Angle from origin
            row['angle_from_origin'] = np.arctan2(pos[1], pos[0])

            # Compute angle to goal (is this intermediate "on the way" to the goal?)
            # Positive = intermediate is in the direction of the goal from agent
            # We use d_goal as proxy for goal direction
            row['is_toward_goal'] = zone_data['d_goal'] < other_data['d_goal']

            rows.append(row)

    return pd.DataFrame(rows)


def analyze_correlations(df):
    """Analyze what features correlate with empirical difficulty."""
    print("\n" + "="*70)
    print("FEATURE CORRELATIONS WITH EMPIRICAL DIFFICULTY")
    print("="*70)

    features = ['d_agent', 'd_goal', 'geometric_total', 'pos_x', 'pos_y',
                'dist_from_origin', 'angle_from_origin']

    print("\nCorrelation with completion_rate:")
    print("-" * 50)
    for feat in features:
        r, p = stats.pearsonr(df[feat], df['completion_rate'])
        sig = "*" if p < 0.05 else " "
        print(f"  {feat:25s}: r = {r:+.3f}  (p = {p:.4f}) {sig}")

    print("\nCorrelation with mean_steps:")
    print("-" * 50)
    for feat in features:
        r, p = stats.pearsonr(df[feat], df['mean_steps'])
        sig = "*" if p < 0.05 else " "
        print(f"  {feat:25s}: r = {r:+.3f}  (p = {p:.4f}) {sig}")


def analyze_choice_predictors(df):
    """Analyze what predicts the agent's choice."""
    print("\n" + "="*70)
    print("WHAT PREDICTS THE AGENT'S CHOICE?")
    print("="*70)

    # Get one row per choice (comparing the two options)
    chosen = df[df['was_chosen'] == True].copy()
    not_chosen = df[df['was_chosen'] == False].copy()

    # Merge on config_id
    merged = pd.merge(
        chosen, not_chosen,
        on='config_id',
        suffixes=('_chosen', '_not_chosen')
    )

    print(f"\nN valid choices: {len(merged)}")

    features = ['d_agent', 'd_goal', 'geometric_total', 'completion_rate', 'pos_x', 'pos_y']

    print("\nMean feature values:")
    print("-" * 60)
    print(f"{'Feature':25s}  {'Chosen':>12s}  {'Not Chosen':>12s}  {'Diff':>10s}  p-value")
    print("-" * 60)

    for feat in features:
        chosen_vals = merged[f'{feat}_chosen']
        not_chosen_vals = merged[f'{feat}_not_chosen']

        mean_chosen = chosen_vals.mean()
        mean_not = not_chosen_vals.mean()

        # Paired t-test
        t, p = stats.ttest_rel(chosen_vals, not_chosen_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {feat:23s}  {mean_chosen:12.3f}  {mean_not:12.3f}  {mean_chosen - mean_not:+10.3f}  {p:.4f} {sig}")

    # Classify choices
    print("\n" + "-"*50)
    print("Choice classification:")

    # Agent chose closer (myopic)
    chose_closer = (merged['d_agent_chosen'] < merged['d_agent_not_chosen']).sum()
    chose_farther = (merged['d_agent_chosen'] > merged['d_agent_not_chosen']).sum()
    print(f"  Chose closer intermediate:    {chose_closer}/{len(merged)} ({100*chose_closer/len(merged):.1f}%)")
    print(f"  Chose farther intermediate:   {chose_farther}/{len(merged)} ({100*chose_farther/len(merged):.1f}%)")

    # Agent chose geometrically better
    chose_geom_better = (merged['geometric_total_chosen'] < merged['geometric_total_not_chosen']).sum()
    print(f"  Chose geometrically better:   {chose_geom_better}/{len(merged)} ({100*chose_geom_better/len(merged):.1f}%)")

    # Agent chose empirically easier
    chose_emp_easier = (merged['completion_rate_chosen'] > merged['completion_rate_not_chosen']).sum()
    print(f"  Chose empirically easier:     {chose_emp_easier}/{len(merged)} ({100*chose_emp_easier/len(merged):.1f}%)")

    return merged


def analyze_disagreement(df):
    """Analyze cases where empirical and geometric difficulty disagree."""
    print("\n" + "="*70)
    print("WHEN DO EMPIRICAL AND GEOMETRIC DISAGREE?")
    print("="*70)

    # Get cases where they disagree
    easier = df[df['is_empirically_easier'] == True].copy()

    agree = easier[easier['is_geometrically_easier'] == True]
    disagree = easier[easier['is_geometrically_easier'] == False]

    print(f"\nEmpirical-geometric agreement: {len(agree)}/{len(easier)} ({100*len(agree)/len(easier):.1f}%)")
    print(f"Disagreement: {len(disagree)}/{len(easier)} ({100*len(disagree)/len(easier):.1f}%)")

    if len(disagree) > 0:
        print("\nWhen they disagree (empirically easier but geometrically harder):")
        print("-" * 50)

        # Compare features in agree vs disagree
        features = ['d_agent', 'd_goal', 'geometric_total', 'completion_rate']

        for feat in features:
            mean_agree = agree[feat].mean()
            mean_disagree = disagree[feat].mean()
            t, p = stats.ttest_ind(agree[feat], disagree[feat])
            sig = "*" if p < 0.05 else " "
            print(f"  {feat:20s}: Agree={mean_agree:.3f}, Disagree={mean_disagree:.3f}, diff={mean_disagree-mean_agree:+.3f} {sig}")

        print("\nKey insight: When they disagree, the empirically easier zone has:")
        print(f"  - Higher d_goal (farther from goal): {disagree['d_goal'].mean():.2f} vs {agree['d_goal'].mean():.2f}")
        print(f"  - But higher completion rate: {disagree['completion_rate'].mean():.2f} vs {agree['completion_rate'].mean():.2f}")


def plot_analysis(df, output_dir='interpretability/results/empirical_difficulty'):
    """Create visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Completion rate vs d_agent
    ax = axes[0, 0]
    ax.scatter(df['d_agent'], df['completion_rate'], alpha=0.5, c=df['was_chosen'].map({True: 'green', False: 'gray'}))
    ax.set_xlabel('Distance from agent to intermediate')
    ax.set_ylabel('Completion rate')
    ax.set_title('Completion rate vs d_agent')
    r, p = stats.pearsonr(df['d_agent'], df['completion_rate'])
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')

    # 2. Completion rate vs d_goal
    ax = axes[0, 1]
    ax.scatter(df['d_goal'], df['completion_rate'], alpha=0.5, c=df['was_chosen'].map({True: 'green', False: 'gray'}))
    ax.set_xlabel('Distance from intermediate to goal')
    ax.set_ylabel('Completion rate')
    ax.set_title('Completion rate vs d_goal')
    r, p = stats.pearsonr(df['d_goal'], df['completion_rate'])
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')

    # 3. Completion rate vs geometric total
    ax = axes[1, 0]
    ax.scatter(df['geometric_total'], df['completion_rate'], alpha=0.5, c=df['was_chosen'].map({True: 'green', False: 'gray'}))
    ax.set_xlabel('Geometric total (d_agent + d_goal)')
    ax.set_ylabel('Completion rate')
    ax.set_title('Completion rate vs geometric total')
    r, p = stats.pearsonr(df['geometric_total'], df['completion_rate'])
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')

    # 4. Zone positions colored by completion rate
    ax = axes[1, 1]
    scatter = ax.scatter(df['pos_x'], df['pos_y'], c=df['completion_rate'],
                         cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='Completion rate')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Zone positions (color = completion rate)')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/empirical_difficulty_analysis.png', dpi=150)
    print(f"\nPlot saved to {output_dir}/empirical_difficulty_analysis.png")

    plt.close()


def analyze_d_goal_hypothesis(df):
    """
    Hypothesis: d_goal dominates empirical difficulty because:
    1. Reaching the intermediate is easy (agent is good at going to closest zone)
    2. The hard part is reaching the goal after the intermediate
    3. So d_goal matters more than d_agent for empirical success
    """
    print("\n" + "="*70)
    print("D_GOAL HYPOTHESIS: Does d_goal dominate empirical difficulty?")
    print("="*70)

    # Multiple regression: completion_rate ~ d_agent + d_goal
    from scipy.stats import pearsonr

    r_agent, p_agent = pearsonr(df['d_agent'], df['completion_rate'])
    r_goal, p_goal = pearsonr(df['d_goal'], df['completion_rate'])
    r_total, p_total = pearsonr(df['geometric_total'], df['completion_rate'])

    print("\nUnivariate correlations with completion rate:")
    print(f"  d_agent:          r = {r_agent:+.3f}  (p = {p_agent:.4f})")
    print(f"  d_goal:           r = {r_goal:+.3f}  (p = {p_goal:.4f})")
    print(f"  geometric_total:  r = {r_total:+.3f}  (p = {p_total:.4f})")

    # Partial correlation: d_goal controlling for d_agent
    from scipy.stats import linregress

    # Residualize d_goal on d_agent
    slope, intercept, _, _, _ = linregress(df['d_agent'], df['d_goal'])
    d_goal_resid = df['d_goal'] - (slope * df['d_agent'] + intercept)

    r_goal_partial, p_goal_partial = pearsonr(d_goal_resid, df['completion_rate'])

    # Residualize d_agent on d_goal
    slope, intercept, _, _, _ = linregress(df['d_goal'], df['d_agent'])
    d_agent_resid = df['d_agent'] - (slope * df['d_goal'] + intercept)

    r_agent_partial, p_agent_partial = pearsonr(d_agent_resid, df['completion_rate'])

    print("\nPartial correlations:")
    print(f"  d_goal | d_agent: r = {r_goal_partial:+.3f}  (p = {p_goal_partial:.4f})")
    print(f"  d_agent | d_goal: r = {r_agent_partial:+.3f}  (p = {p_agent_partial:.4f})")

    print("\nInterpretation:")
    if abs(r_goal_partial) > abs(r_agent_partial):
        print("  d_goal has stronger unique contribution to empirical difficulty")
    else:
        print("  d_agent has stronger unique contribution to empirical difficulty")


def analyze_goal_direction(df):
    """Analyze if the agent prefers intermediates toward the goal."""
    print("\n" + "="*70)
    print("GOAL DIRECTION ANALYSIS")
    print("="*70)

    # Get one row per choice - merge on config_id to ensure matching
    chosen = df[df['was_chosen'] == True].copy()
    not_chosen = df[df['was_chosen'] == False].copy()

    merged = pd.merge(
        chosen, not_chosen,
        on='config_id',
        suffixes=('_chosen', '_notchosen')
    )

    # Did agent choose the one closer to goal?
    chose_toward_goal = (merged['d_goal_chosen'] < merged['d_goal_notchosen']).sum()
    total = len(merged)
    print(f"\nAgent chose intermediate CLOSER to goal: {chose_toward_goal}/{total} ({100*chose_toward_goal/total:.1f}%)")

    # Completion rate for toward-goal vs away-from-goal
    toward_goal = df[df['is_toward_goal'] == True]
    away_from_goal = df[df['is_toward_goal'] == False]
    print(f"\nCompletion rate by goal direction:")
    print(f"  Toward goal (smaller d_goal):   {toward_goal['completion_rate'].mean():.3f}")
    print(f"  Away from goal (larger d_goal): {away_from_goal['completion_rate'].mean():.3f}")

    # Is there a confound between pos_x and toward_goal?
    print(f"\nIs 'toward goal' confounded with spatial position?")
    toward_left = (toward_goal['pos_x'] < 0).sum() / len(toward_goal)
    away_left = (away_from_goal['pos_x'] < 0).sum() / len(away_from_goal)
    print(f"  Toward-goal zones on LEFT:  {toward_left*100:.1f}%")
    print(f"  Away-from-goal zones on LEFT: {away_left*100:.1f}%")

    # Correlation between d_goal and pos_x
    r, p = stats.pearsonr(df['d_goal'], df['pos_x'])
    print(f"\nCorrelation d_goal ~ pos_x: r = {r:.3f} (p = {p:.4f})")

    # Multiple regression: choice ~ d_goal + pos_x
    print("\nMultiple logistic regression: choice ~ d_goal + pos_x")
    from scipy.stats import pearsonr

    # Simple approach: partial correlations
    # Choice vs d_goal controlling for pos_x
    slope, intercept, _, _, _ = stats.linregress(df['pos_x'], df['d_goal'])
    d_goal_resid = df['d_goal'] - (slope * df['pos_x'] + intercept)
    r_goal_partial, p_goal_partial = pearsonr(d_goal_resid, df['was_chosen'].astype(float))

    slope, intercept, _, _, _ = stats.linregress(df['d_goal'], df['pos_x'])
    pos_x_resid = df['pos_x'] - (slope * df['d_goal'] + intercept)
    r_posx_partial, p_posx_partial = pearsonr(pos_x_resid, df['was_chosen'].astype(float))

    print(f"  Choice ~ d_goal | pos_x:  r = {r_goal_partial:+.3f} (p = {p_goal_partial:.4f})")
    print(f"  Choice ~ pos_x | d_goal:  r = {r_posx_partial:+.3f} (p = {p_posx_partial:.4f})")


def analyze_spatial_and_color_biases(df):
    """Analyze spatial and color biases in agent choice."""
    print("\n" + "="*70)
    print("SPATIAL AND COLOR BIASES")
    print("="*70)

    # Get choice data
    chosen = df[df['was_chosen'] == True].copy()

    # Spatial bias
    print("\nSpatial bias in agent choice:")
    print("-" * 50)
    chose_left = (chosen['pos_x'] < 0).sum()
    chose_right = (chosen['pos_x'] > 0).sum()
    total = len(chosen)
    print(f"  Chose zone on LEFT (x < 0):   {chose_left}/{total} ({100*chose_left/total:.1f}%)")
    print(f"  Chose zone on RIGHT (x > 0):  {chose_right}/{total} ({100*chose_right/total:.1f}%)")

    chose_top = (chosen['pos_y'] > 0).sum()
    chose_bottom = (chosen['pos_y'] < 0).sum()
    print(f"  Chose zone on TOP (y > 0):    {chose_top}/{total} ({100*chose_top/total:.1f}%)")
    print(f"  Chose zone on BOTTOM (y < 0): {chose_bottom}/{total} ({100*chose_bottom/total:.1f}%)")

    # Quadrant analysis
    print("\nQuadrant breakdown (chosen zones):")
    q1 = ((chosen['pos_x'] > 0) & (chosen['pos_y'] > 0)).sum()  # Top-right
    q2 = ((chosen['pos_x'] < 0) & (chosen['pos_y'] > 0)).sum()  # Top-left
    q3 = ((chosen['pos_x'] < 0) & (chosen['pos_y'] < 0)).sum()  # Bottom-left
    q4 = ((chosen['pos_x'] > 0) & (chosen['pos_y'] < 0)).sum()  # Bottom-right
    print(f"  Q1 (top-right, +x +y):    {q1}/{total} ({100*q1/total:.1f}%)")
    print(f"  Q2 (top-left, -x +y):     {q2}/{total} ({100*q2/total:.1f}%)")
    print(f"  Q3 (bottom-left, -x -y):  {q3}/{total} ({100*q3/total:.1f}%)")
    print(f"  Q4 (bottom-right, +x -y): {q4}/{total} ({100*q4/total:.1f}%)")

    # Color bias (if int_color varies)
    if 'int_color' in df.columns:
        print("\nColor bias:")
        print("-" * 50)
        color_choices = chosen.groupby('int_color').size()
        for color, count in color_choices.items():
            n_configs = df[df['int_color'] == color]['config_id'].nunique()
            print(f"  {color}: {count} choices from {n_configs} configs")

    # Completion rate by quadrant
    print("\nCompletion rate by zone position:")
    print("-" * 50)
    left_zones = df[df['pos_x'] < 0]
    right_zones = df[df['pos_x'] > 0]
    print(f"  LEFT zones (x < 0):  mean completion = {left_zones['completion_rate'].mean():.3f}")
    print(f"  RIGHT zones (x > 0): mean completion = {right_zones['completion_rate'].mean():.3f}")

    top_zones = df[df['pos_y'] > 0]
    bottom_zones = df[df['pos_y'] < 0]
    print(f"  TOP zones (y > 0):   mean completion = {top_zones['completion_rate'].mean():.3f}")
    print(f"  BOTTOM zones (y < 0): mean completion = {bottom_zones['completion_rate'].mean():.3f}")


def plot_spatial_bias(df, output_dir='interpretability/results/empirical_difficulty'):
    """Plot spatial distribution of chosen vs not-chosen zones."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Chosen vs not chosen positions
    ax = axes[0]
    chosen = df[df['was_chosen'] == True]
    not_chosen = df[df['was_chosen'] == False]
    ax.scatter(not_chosen['pos_x'], not_chosen['pos_y'], c='gray', alpha=0.5, label='Not chosen', s=50)
    ax.scatter(chosen['pos_x'], chosen['pos_y'], c='green', alpha=0.7, label='Chosen', s=80)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Agent zone choice spatial distribution')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')

    # 2. Histogram of chosen x positions
    ax = axes[1]
    ax.hist(chosen['pos_x'], bins=15, alpha=0.7, label='Chosen', color='green')
    ax.hist(not_chosen['pos_x'], bins=15, alpha=0.5, label='Not chosen', color='gray')
    ax.set_xlabel('X position')
    ax.set_ylabel('Count')
    ax.set_title('X position distribution')
    ax.legend()
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)

    # 3. Completion rate heatmap
    ax = axes[2]
    # Bin positions and compute mean completion rate
    x_bins = np.linspace(-2.5, 2.5, 10)
    y_bins = np.linspace(-2.5, 2.5, 10)
    completion_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    count_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))

    for _, row in df.iterrows():
        xi = np.searchsorted(x_bins, row['pos_x']) - 1
        yi = np.searchsorted(y_bins, row['pos_y']) - 1
        if 0 <= xi < len(x_bins)-1 and 0 <= yi < len(y_bins)-1:
            completion_grid[yi, xi] += row['completion_rate']
            count_grid[yi, xi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_completion = completion_grid / count_grid

    im = ax.imshow(mean_completion, origin='lower', extent=[-2.5, 2.5, -2.5, 2.5],
                   cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean completion rate')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Completion rate by position')
    ax.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/spatial_bias_analysis.png', dpi=150)
    print(f"\nSpatial bias plot saved to {output_dir}/spatial_bias_analysis.png")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='fresh_baseline')
    parser.add_argument('--test', default='optvar', choices=['optvar', 'opteq'])
    args = parser.parse_args()

    # Find most recent results file
    import glob
    pattern = f'interpretability/results/empirical_difficulty/empirical_difficulty_{args.exp}_*.json'
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No results found for pattern: {pattern}")
        return

    filepath = files[-1]  # Most recent
    print(f"Loading: {filepath}")

    results = load_results(filepath)
    df = extract_features(results, args.test)

    print(f"\nLoaded {len(df)} zone observations from {len(df)//2} configurations")

    # Run analyses
    analyze_correlations(df)
    merged = analyze_choice_predictors(df)
    analyze_disagreement(df)
    analyze_d_goal_hypothesis(df)
    analyze_goal_direction(df)
    analyze_spatial_and_color_biases(df)
    plot_analysis(df)
    plot_spatial_bias(df)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)


if __name__ == '__main__':
    main()
