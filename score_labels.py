"""Compare your hand labels with the judge.

Usage:  python score_labels.py LABEL_ME_hand_labels.csv KEY_do_not_open_until_labeled.csv
Run this only after you have filled in your_verdict for every row.
"""
import sys
import pandas as pd

labels_path, key_path = sys.argv[1], sys.argv[2]
labels = pd.read_csv(labels_path)
key = pd.read_csv(key_path)
merged = labels.merge(key, on='sample_id')

v = merged['your_verdict (PASS or FAIL)'].fillna('').str.strip().str.upper()
if not v.isin(['PASS', 'FAIL']).all():
    bad = merged.loc[~v.isin(['PASS', 'FAIL']), 'sample_id'].tolist()
    sys.exit(f"Missing or invalid verdict (use PASS or FAIL) for: {bad}")

merged['human_passed'] = v.eq('PASS')
merged['judge_passed'] = merged['judge_passed'].astype(bool)
agree = merged['human_passed'] == merged['judge_passed']

n = len(merged)
p_o = agree.mean()
# Cohen's kappa: agreement beyond chance, for a binary label
p_h = merged['human_passed'].mean(); p_j = merged['judge_passed'].mean()
p_e = p_h * p_j + (1 - p_h) * (1 - p_j)
kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else float('nan')

print(f"Items labelled: {n}")
print(f"Agreement with judge: {agree.sum()}/{n} = {p_o:.0%}")
print(f"Cohen's kappa: {kappa:.2f}  (0.61-0.80 substantial, above 0.80 almost perfect)")
print()
print("Disagreements:")
dis = merged[~agree]
if dis.empty:
    print("  none")
else:
    print(dis[['sample_id','label','judge_passed','human_passed','judge_reason','your_note (optional)']]
          .to_string(index=False, max_colwidth=70))
