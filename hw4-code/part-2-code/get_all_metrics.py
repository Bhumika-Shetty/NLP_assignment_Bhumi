#!/usr/bin/env python3
"""
Get metrics for all experiments to identify the best model for Q6.
"""
from utils import compute_metrics
import os

# Ground truth paths
gt_sql = "data/dev.sql"
gt_records = "records/ground_truth_dev.pkl"

# List of experiments to evaluate
experiments = [
    "t5_ft_experiment_d",
    "t5_ft_final_sub",
    "t5_ft_final",
    "t5_ft_baseline_6epoch",
    "t5_ft_arch_no_decay",
    "t5_ft_arch_high_decay",
    "t5_ft_base",
]

print("=" * 80)
print("Evaluating All Experiments - Dev Set Results")
print("=" * 80)
print(f"{'Experiment':<30} {'Query EM':<12} {'Record EM':<12} {'F1 Score':<12} {'Error Rate':<12}")
print("-" * 80)

results = []

for exp in experiments:
    pred_sql = f"results/{exp}_dev.sql"
    pred_records = f"records/{exp}_dev.pkl"
    
    if os.path.exists(pred_sql) and os.path.exists(pred_records):
        try:
            sql_em, record_em, record_f1, error_msgs = compute_metrics(
                gt_sql, pred_sql, gt_records, pred_records
            )
            num_errors = sum(1 for msg in error_msgs if msg != "")
            error_rate = num_errors / len(error_msgs) if error_msgs else 0
            
            results.append({
                'name': exp,
                'sql_em': sql_em,
                'record_em': record_em,
                'f1': record_f1,
                'error_rate': error_rate
            })
            
            print(f"{exp:<30} {sql_em*100:>10.2f}% {record_em*100:>10.2f}% {record_f1*100:>10.2f}% {error_rate*100:>10.2f}%")
        except Exception as e:
            print(f"{exp:<30} ERROR: {str(e)}")

print("\n" + "=" * 80)
print("Best Model (by F1 Score):")
print("=" * 80)
if results:
    best = max(results, key=lambda x: x['f1'])
    print(f"Experiment: {best['name']}")
    print(f"  Query EM: {best['sql_em']*100:.2f}%")
    print(f"  Record EM: {best['record_em']*100:.2f}%")
    print(f"  F1 Score: {best['f1']*100:.2f}%")
    print(f"  Error Rate: {best['error_rate']*100:.2f}%")

