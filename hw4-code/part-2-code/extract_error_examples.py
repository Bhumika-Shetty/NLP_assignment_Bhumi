#!/usr/bin/env python3
"""
Extract detailed error examples for Table 5.
"""
import pickle
import re

# Paths for t5_ft_final model
pred_sql = "results/t5_ft_final_dev.sql"
pred_records = "records/t5_ft_final_dev.pkl"
gt_sql = "data/dev.sql"
gt_records = "records/ground_truth_dev.pkl"

def load_sql_queries(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_records(path):
    with open(path, 'rb') as f:
        records, error_msgs = pickle.load(f)
    return records, error_msgs

def load_nl_queries(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

print("=" * 80)
print("DETAILED ERROR EXAMPLES FOR TABLE 5")
print("=" * 80)

gt_queries = load_sql_queries(gt_sql)
pred_queries = load_sql_queries(pred_sql)
gt_recs, _ = load_records(gt_records)
pred_recs, error_msgs = load_records(pred_records)
nl_queries = load_nl_queries("data/dev.nl")

total = len(gt_queries)

# 1. SQL Syntax Errors - Missing Comparison Operators
print("\n" + "=" * 80)
print("1. MISSING COMPARISON OPERATORS IN WHERE CLAUSES")
print("=" * 80)
missing_ops = []
for i, (error_msg, pred_q, gt_q, nl) in enumerate(zip(error_msgs, pred_queries, gt_queries, nl_queries)):
    if error_msg != "":
        # Check for patterns like "column value" without operator
        if re.search(r'\b\w+\.\w+\s+\d+\b', pred_q) and not re.search(r'[<>=!]', pred_q.split('WHERE')[-1] if 'WHERE' in pred_q else ''):
            missing_ops.append((i, error_msg, pred_q, gt_q, nl))
        elif 'syntax error' in error_msg.lower() and any(op in error_msg for op in ['near', 'operator']):
            missing_ops.append((i, error_msg, pred_q, gt_q, nl))

print(f"Total: {len(missing_ops)}/{total} ({len(missing_ops)/total:.1%})")
for i, (idx, err, pred, gt, nl) in enumerate(missing_ops[:2]):
    print(f"\nExample {i+1}:")
    print(f"NL Query: {nl}")
    print(f"Predicted SQL: {pred}")
    print(f"Ground Truth SQL: {gt}")
    print(f"Error: {err[:200]}...")

# 2. Record Mismatch - Incorrect WHERE Conditions
print("\n" + "=" * 80)
print("2. INCORRECT WHERE CONDITIONS (Wrong Filters)")
print("=" * 80)
incorrect_where = []
for i, (pred_rec, gt_rec, pred_q, gt_q, nl, err) in enumerate(zip(pred_recs, gt_recs, pred_queries, gt_queries, nl_queries, error_msgs)):
    if err == "" and set(pred_rec) != set(gt_rec):
        # Check if it's a WHERE condition issue (not just column selection)
        if "WHERE" in pred_q.upper() and "WHERE" in gt_q.upper():
            incorrect_where.append((i, pred_rec, gt_rec, pred_q, gt_q, nl))

print(f"Total: {len(incorrect_where)}/{total} ({len(incorrect_where)/total:.1%})")
for i, (idx, pred_r, gt_r, pred, gt, nl) in enumerate(incorrect_where[:2]):
    print(f"\nExample {i+1}:")
    print(f"NL Query: {nl}")
    print(f"Predicted SQL: {pred}")
    print(f"Ground Truth SQL: {gt}")
    print(f"Predicted Records: {len(pred_r)} records")
    print(f"Ground Truth Records: {len(gt_r)} records")

# 3. Missing WHERE Conditions
print("\n" + "=" * 80)
print("3. MISSING OR INCORRECT WHERE CLAUSE CONDITIONS")
print("=" * 80)
missing_where_cond = []
for i, (pred_rec, gt_rec, pred_q, gt_q, nl, err) in enumerate(zip(pred_recs, gt_recs, pred_queries, gt_queries, nl_queries, error_msgs)):
    if err == "" and set(pred_rec) != set(gt_rec):
        # Check if WHERE clause is missing conditions
        if "WHERE" in gt_q.upper() and "WHERE" in pred_q.upper():
            # Count conditions (rough heuristic)
            gt_conditions = len(re.findall(r'\bAND\b|\bOR\b', gt_q, re.IGNORECASE)) + 1
            pred_conditions = len(re.findall(r'\bAND\b|\bOR\b', pred_q, re.IGNORECASE)) + 1
            if gt_conditions > pred_conditions:
                missing_where_cond.append((i, pred_rec, gt_rec, pred_q, gt_q, nl))

print(f"Total: {len(missing_where_cond)}/{total} ({len(missing_where_cond)/total:.1%})")
for i, (idx, pred_r, gt_r, pred, gt, nl) in enumerate(missing_where_cond[:2]):
    print(f"\nExample {i+1}:")
    print(f"NL Query: {nl}")
    print(f"Predicted SQL: {pred}")
    print(f"Ground Truth SQL: {gt}")
    print(f"Predicted Records: {len(pred_r)} records")
    print(f"Ground Truth Records: {len(gt_r)} records")

# 4. Missing Columns in SELECT
print("\n" + "=" * 80)
print("4. MISSING COLUMNS IN SELECT CLAUSE")
print("=" * 80)
missing_cols = []
for i, (pred_q, gt_q, nl, err) in enumerate(zip(pred_queries, gt_queries, nl_queries, error_msgs)):
    if err == "":
        pred_select = re.search(r'SELECT\s+(.+?)\s+FROM', pred_q, re.IGNORECASE | re.DOTALL)
        gt_select = re.search(r'SELECT\s+(.+?)\s+FROM', gt_q, re.IGNORECASE | re.DOTALL)
        if pred_select and gt_select:
            pred_cols_set = set([c.strip().upper() for c in pred_select.group(1).split(',')])
            gt_cols_set = set([c.strip().upper() for c in gt_select.group(1).split(',')])
            if gt_cols_set - pred_cols_set:  # Missing columns
                missing_cols.append((i, pred_q, gt_q, nl))

print(f"Total: {len(missing_cols)}/{total} ({len(missing_cols)/total:.1%})")
for i, (idx, pred, gt, nl) in enumerate(missing_cols[:2]):
    print(f"\nExample {i+1}:")
    print(f"NL Query: {nl}")
    print(f"Predicted SQL: {pred}")
    print(f"Ground Truth SQL: {gt}")

print("\n" + "=" * 80)
print("Use these examples to fill Table 5")
print("=" * 80)

