#!/usr/bin/env python3
"""
Standalone script to analyze errors in dev set predictions for Table 5.
Reads files directly without heavy dependencies.
"""
import pickle
import re

# Paths for t5_ft_final model
pred_sql = "results/t5_ft_final_dev.sql"
pred_records = "records/t5_ft_final_dev.pkl"
gt_sql = "data/dev.sql"
gt_records = "records/ground_truth_dev.pkl"

# Load SQL queries
def load_sql_queries(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load records
def load_records(path):
    with open(path, 'rb') as f:
        records, error_msgs = pickle.load(f)
    return records, error_msgs

# Load NL queries
def load_nl_queries(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

print("=" * 80)
print("Loading data...")
print("=" * 80)

gt_queries = load_sql_queries(gt_sql)
pred_queries = load_sql_queries(pred_sql)
gt_recs, _ = load_records(gt_records)
pred_recs, error_msgs = load_records(pred_records)
nl_queries = load_nl_queries("data/dev.nl")

print(f"Loaded {len(gt_queries)} queries\n")

# 1. SQL Syntax Errors
print("=" * 80)
print("1. SQL SYNTAX ERRORS")
print("=" * 80)
syntax_errors = []
for i, (error_msg, pred_q, gt_q, nl) in enumerate(zip(error_msgs, pred_queries, gt_queries, nl_queries)):
    if error_msg != "":
        syntax_errors.append((i, error_msg, pred_q, gt_q, nl))

print(f"Total SQL syntax errors: {len(syntax_errors)}/{len(error_msgs)} ({len(syntax_errors)/len(error_msgs)*100:.1f}%)")
if syntax_errors:
    print("\nExample 1 - SQL Syntax Error:")
    idx, err, pred, gt, nl = syntax_errors[0]
    print(f"  NL Query: {nl}")
    print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
    print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")
    print(f"  Error Message: {err[:150]}...")
    
    if len(syntax_errors) > 1:
        print("\nExample 2 - SQL Syntax Error:")
        idx, err, pred, gt, nl = syntax_errors[1]
        print(f"  NL Query: {nl}")
        print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
        print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")
        print(f"  Error Message: {err[:150]}...")

# 2. Record Mismatch (but SQL executes correctly)
print("\n" + "=" * 80)
print("2. RECORD MISMATCH (SQL executes but wrong results)")
print("=" * 80)
record_mismatches = []
for i, (pred_rec, gt_rec, pred_q, gt_q, nl, err) in enumerate(zip(pred_recs, gt_recs, pred_queries, gt_queries, nl_queries, error_msgs)):
    if err == "" and set(pred_rec) != set(gt_rec):
        record_mismatches.append((i, pred_rec, gt_rec, pred_q, gt_q, nl))

print(f"Total record mismatches: {len(record_mismatches)}/{len(gt_queries)} ({len(record_mismatches)/len(gt_queries)*100:.1f}%)")
if record_mismatches:
    print("\nExample 1 - Record Mismatch:")
    idx, pred_r, gt_r, pred, gt, nl = record_mismatches[0]
    print(f"  NL Query: {nl}")
    print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
    print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")
    print(f"  Predicted Records Count: {len(pred_r)}")
    print(f"  Ground Truth Records Count: {len(gt_r)}")
    print(f"  Predicted Records (first 3): {pred_r[:3] if len(pred_r) > 3 else pred_r}")
    print(f"  Ground Truth Records (first 3): {gt_r[:3] if len(gt_r) > 3 else gt_r}")

# 3. Missing/Incorrect WHERE clauses
print("\n" + "=" * 80)
print("3. MISSING OR INCORRECT WHERE CLAUSES")
print("=" * 80)
where_errors = []
for i, (pred_q, gt_q, nl, err) in enumerate(zip(pred_queries, gt_queries, nl_queries, error_msgs)):
    pred_has_where = "WHERE" in pred_q.upper()
    gt_has_where = "WHERE" in gt_q.upper()
    
    # Check if WHERE clause is missing or different
    if pred_has_where != gt_has_where:
        where_errors.append((i, pred_q, gt_q, nl, "Missing WHERE" if not pred_has_where else "Extra WHERE"))
    elif pred_has_where and gt_has_where:
        # Extract WHERE clauses
        pred_where = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', pred_q, re.IGNORECASE | re.DOTALL)
        gt_where = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', gt_q, re.IGNORECASE | re.DOTALL)
        if pred_where and gt_where:
            pred_where_clean = pred_where.group(1).strip().upper().replace(' ', '')
            gt_where_clean = gt_where.group(1).strip().upper().replace(' ', '')
            if pred_where_clean != gt_where_clean:
                where_errors.append((i, pred_q, gt_q, nl, "Incorrect WHERE condition"))

print(f"Total WHERE clause errors: {len(where_errors)}/{len(gt_queries)} ({len(where_errors)/len(gt_queries)*100:.1f}%)")
if where_errors:
    print("\nExample 1 - WHERE Clause Error:")
    idx, pred, gt, nl, err_type = where_errors[0]
    print(f"  Error Type: {err_type}")
    print(f"  NL Query: {nl}")
    print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
    print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")

# 4. Incorrect JOIN operations
print("\n" + "=" * 80)
print("4. INCORRECT JOIN OPERATIONS")
print("=" * 80)
join_errors = []
for i, (pred_q, gt_q, nl, err) in enumerate(zip(pred_queries, gt_queries, nl_queries, error_msgs)):
    pred_joins = len(re.findall(r'\bJOIN\b', pred_q, re.IGNORECASE))
    gt_joins = len(re.findall(r'\bJOIN\b', gt_q, re.IGNORECASE))
    
    if pred_joins != gt_joins:
        join_errors.append((i, pred_q, gt_q, nl, f"JOIN count mismatch: {pred_joins} vs {gt_joins}"))
    elif pred_joins > 0 and gt_joins > 0:
        # Check if JOIN structure is different
        pred_join_tables = re.findall(r'JOIN\s+(\w+)', pred_q, re.IGNORECASE)
        gt_join_tables = re.findall(r'JOIN\s+(\w+)', gt_q, re.IGNORECASE)
        if set(pred_join_tables) != set(gt_join_tables):
            join_errors.append((i, pred_q, gt_q, nl, f"Different JOIN tables: {pred_join_tables} vs {gt_join_tables}"))

print(f"Total JOIN errors: {len(join_errors)}/{len(gt_queries)} ({len(join_errors)/len(gt_queries)*100:.1f}%)")
if join_errors:
    print("\nExample 1 - JOIN Error:")
    idx, pred, gt, nl, err_type = join_errors[0]
    print(f"  Error Type: {err_type}")
    print(f"  NL Query: {nl}")
    print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
    print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")

# 5. Column name errors (simplified check)
print("\n" + "=" * 80)
print("5. COLUMN/SELECT CLAUSE ERRORS")
print("=" * 80)
column_errors = []
for i, (pred_q, gt_q, nl, err) in enumerate(zip(pred_queries, gt_queries, nl_queries, error_msgs)):
    # Extract SELECT clauses
    pred_select = re.search(r'SELECT\s+(.+?)\s+FROM', pred_q, re.IGNORECASE | re.DOTALL)
    gt_select = re.search(r'SELECT\s+(.+?)\s+FROM', gt_q, re.IGNORECASE | re.DOTALL)
    
    if pred_select and gt_select:
        pred_cols = pred_select.group(1).strip()
        gt_cols = gt_select.group(1).strip()
        # Simple check - if very different (not just order)
        if pred_cols.upper() != gt_cols.upper() and "*" not in pred_cols and "*" not in gt_cols:
            # Check if it's a significant difference (not just spacing)
            pred_cols_set = set([c.strip().upper() for c in pred_cols.split(',')])
            gt_cols_set = set([c.strip().upper() for c in gt_cols.split(',')])
            if pred_cols_set != gt_cols_set:
                column_errors.append((i, pred_q, gt_q, nl))

print(f"Total column/SELECT errors: {len(column_errors)}/{len(gt_queries)} ({len(column_errors)/len(gt_queries)*100:.1f}%)")
if column_errors:
    print("\nExample 1 - Column/SELECT Error:")
    idx, pred, gt, nl = column_errors[0]
    print(f"  NL Query: {nl}")
    print(f"  Predicted SQL: {pred[:200]}..." if len(pred) > 200 else f"  Predicted SQL: {pred}")
    print(f"  Ground Truth SQL: {gt[:200]}..." if len(gt) > 200 else f"  Ground Truth SQL: {gt}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY FOR TABLE 5")
print("=" * 80)
print(f"1. SQL Syntax Errors: {len(syntax_errors)}/{len(error_msgs)} ({len(syntax_errors)/len(error_msgs)*100:.1f}%)")
print(f"2. Record Mismatches: {len(record_mismatches)}/{len(gt_queries)} ({len(record_mismatches)/len(gt_queries)*100:.1f}%)")
print(f"3. WHERE Clause Errors: {len(where_errors)}/{len(gt_queries)} ({len(where_errors)/len(gt_queries)*100:.1f}%)")
print(f"4. JOIN Errors: {len(join_errors)}/{len(gt_queries)} ({len(join_errors)/len(gt_queries)*100:.1f}%)")
print(f"5. Column/SELECT Errors: {len(column_errors)}/{len(gt_queries)} ({len(column_errors)/len(gt_queries)*100:.1f}%)")
print("=" * 80)
print("\nNote: Review the examples above to extract specific snippets for Table 5.")

