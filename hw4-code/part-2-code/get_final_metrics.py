#!/usr/bin/env python3
"""
Get metrics for the final model (t5_ft_final) for Q6 Table 4.
"""
import os
from utils import compute_metrics

# Final model paths
model_name = "t5_ft_final"

# Dev set evaluation
print("=" * 80)
print("Development Set Results - t5_ft_final")
print("=" * 80)
dev_pred_sql = f"results/{model_name}_dev.sql"
dev_pred_records = f"records/{model_name}_dev.pkl"
dev_gt_sql = "data/dev.sql"
dev_gt_records = "records/ground_truth_dev.pkl"

sql_em, record_em, record_f1, error_msgs = compute_metrics(
    dev_gt_sql, dev_pred_sql, dev_gt_records, dev_pred_records
)
num_errors = sum(1 for msg in error_msgs if msg != "")
error_rate = num_errors / len(error_msgs) if error_msgs else 0

print(f"Query EM:     {sql_em*100:.2f}% ({sql_em:.4f})")
print(f"Record EM:    {record_em*100:.2f}% ({record_em:.4f})")
print(f"F1 Score:     {record_f1*100:.2f}% ({record_f1:.4f})")
print(f"Error Rate:   {error_rate*100:.2f}% ({num_errors}/{len(error_msgs)})")

# Test set evaluation (if available)
print("\n" + "=" * 80)
print("Test Set Results - t5_ft_final")
print("=" * 80)
test_pred_sql = f"results/{model_name}_test.sql"
test_pred_records = f"records/{model_name}_test.pkl"

if os.path.exists(test_pred_sql) and os.path.exists(test_pred_records):
    # Note: We don't have test ground truth, so we can't compute metrics locally
    # This would need to come from Gradescope
    print("Test files exist but ground truth not available locally.")
    print("Test metrics should be obtained from Gradescope leaderboard.")
    print(f"Files ready for submission:")
    print(f"  {test_pred_sql}")
    print(f"  {test_pred_records}")
else:
    print("Test files not found.")

print("\n" + "=" * 80)
print("For Table 4:")
print("=" * 80)
print(f"Dev Results - Query EM: {sql_em*100:.2f}%, F1: {record_f1*100:.2f}%")
print("Test Results - Get from Gradescope leaderboard")

