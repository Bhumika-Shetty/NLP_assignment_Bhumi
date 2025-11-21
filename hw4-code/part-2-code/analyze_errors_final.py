#!/usr/bin/env python3
"""
Error analysis for t5_ft_final model - Q6 Table 5
"""
import os
import pickle
import sqlite3
import re
from collections import Counter
from typing import List, Tuple

DB_PATH = 'data/flight_database.db'

def read_queries(sql_path: str):
    with open(sql_path, 'r') as f:
        qs = [q.strip() for q in f.readlines()]
    return qs

def load_queries_and_records(sql_path: str, record_path: str):
    read_qs = read_queries(sql_path)
    if record_path is not None and os.path.exists(record_path):
        with open(record_path, 'rb') as f:
            records, error_msgs = pickle.load(f)
    else:
        records = [[] for _ in read_qs]
        error_msgs = ["Records not found" for _ in read_qs]
    return read_qs, records, error_msgs

def analyze_errors(model_name, gt_sql_path, model_sql_path, gt_record_path, model_record_path, nl_path):
    print("=" * 80)
    print(f"ERROR ANALYSIS FOR {model_name.upper()}")
    print("=" * 80)

    gt_nl_queries = read_queries(nl_path)
    gt_sql_queries, gt_records, _ = load_queries_and_records(gt_sql_path, gt_record_path)
    model_sql_queries, model_records, model_error_msgs = load_queries_and_records(model_sql_path, model_record_path)

    total = len(gt_sql_queries)
    print(f"Total queries: {total}\n")

    # 1. SQL Syntax Errors
    syntax_errors = [(i, msg, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i])
                     for i, msg in enumerate(model_error_msgs) if msg != ""]
    
    print("=" * 80)
    print(f"1. SQL SYNTAX ERRORS: {len(syntax_errors)}/{total} ({len(syntax_errors)/total:.1%})")
    print("=" * 80)
    for i, (idx, msg, pred_sql, gt_sql, nl_query) in enumerate(syntax_errors[:3]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")
        print(f"Error: {msg[:150]}...")

    # 2. Record Mismatch (SQL executes but wrong results)
    record_mismatches = []
    for i in range(len(gt_sql_queries)):
        if model_error_msgs[i] == "" and set(gt_records[i]) != set(model_records[i]):
            record_mismatches.append((i, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i], 
                                     model_records[i], gt_records[i]))
    
    print("\n" + "=" * 80)
    print(f"2. RECORD MISMATCHES: {len(record_mismatches)}/{total} ({len(record_mismatches)/total:.1%})")
    print("=" * 80)
    for i, (idx, pred_sql, gt_sql, nl_query, pred_rec, gt_rec) in enumerate(record_mismatches[:3]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")
        print(f"Predicted Records: {len(pred_rec)} records")
        print(f"Ground Truth Records: {len(gt_rec)} records")

    # 3. Missing WHERE Clauses
    missing_where = []
    for i in range(len(gt_sql_queries)):
        if model_error_msgs[i] == "" and set(gt_records[i]) != set(model_records[i]):
            if "WHERE" in gt_sql_queries[i].upper() and "WHERE" not in model_sql_queries[i].upper():
                missing_where.append((i, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i]))
    
    print("\n" + "=" * 80)
    print(f"3. MISSING WHERE CLAUSES: {len(missing_where)}/{total} ({len(missing_where)/total:.1%})")
    print("=" * 80)
    for i, (idx, pred_sql, gt_sql, nl_query) in enumerate(missing_where[:2]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")

    # 4. Incorrect WHERE Conditions
    incorrect_where = []
    for i in range(len(gt_sql_queries)):
        if model_error_msgs[i] == "" and set(gt_records[i]) != set(model_records[i]):
            if "WHERE" in gt_sql_queries[i].upper() and "WHERE" in model_sql_queries[i].upper():
                gt_where = gt_sql_queries[i].upper().split("WHERE", 1)[1] if "WHERE" in gt_sql_queries[i].upper() else ""
                pred_where = model_sql_queries[i].upper().split("WHERE", 1)[1] if "WHERE" in model_sql_queries[i].upper() else ""
                if len(gt_where) > 0 and len(pred_where) > 0:
                    # Check if WHERE clauses are significantly different
                    if gt_where not in pred_where and pred_where not in gt_where:
                        # Additional check: are they semantically different?
                        incorrect_where.append((i, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i]))
    
    print("\n" + "=" * 80)
    print(f"4. INCORRECT WHERE CONDITIONS: {len(incorrect_where)}/{total} ({len(incorrect_where)/total:.1%})")
    print("=" * 80)
    for i, (idx, pred_sql, gt_sql, nl_query) in enumerate(incorrect_where[:2]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")

    # 5. Missing/Incorrect SELECT Columns
    select_errors = []
    for i in range(len(gt_sql_queries)):
        if model_error_msgs[i] == "" and set(gt_records[i]) != set(model_records[i]):
            gt_select = re.findall(r'SELECT\s+(.*?)\s+FROM', gt_sql_queries[i].upper())
            pred_select = re.findall(r'SELECT\s+(.*?)\s+FROM', model_sql_queries[i].upper())
            if gt_select and pred_select and gt_select[0] != pred_select[0]:
                select_errors.append((i, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i]))
    
    print("\n" + "=" * 80)
    print(f"5. MISSING/INCORRECT SELECT COLUMNS: {len(select_errors)}/{total} ({len(select_errors)/total:.1%})")
    print("=" * 80)
    for i, (idx, pred_sql, gt_sql, nl_query) in enumerate(select_errors[:2]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")

    # 6. Missing Comparison Operators
    missing_ops = []
    for i, msg in enumerate(model_error_msgs):
        if msg != "":
            # Check for common operator-related errors
            if any(op in msg.upper() for op in ['OPERATOR', 'SYNTAX', 'NEAR', 'EXPECTED']):
                # Check if predicted SQL is missing operators
                pred_sql = model_sql_queries[i]
                gt_sql = gt_sql_queries[i]
                # Look for patterns like "column value" instead of "column = value" or "column < value"
                if re.search(r'\b\w+\.\w+\s+\d+\b', pred_sql) and not re.search(r'[<>=]', pred_sql.split('WHERE')[-1] if 'WHERE' in pred_sql else ''):
                    missing_ops.append((i, model_sql_queries[i], gt_sql_queries[i], gt_nl_queries[i], msg))
    
    print("\n" + "=" * 80)
    print(f"6. MISSING COMPARISON OPERATORS: {len(missing_ops)}/{total} ({len(missing_ops)/total:.1%})")
    print("=" * 80)
    for i, (idx, pred_sql, gt_sql, nl_query, msg) in enumerate(missing_ops[:2]):
        print(f"\nExample {i+1}:")
        print(f"NL Query: {nl_query}")
        print(f"Predicted SQL: {pred_sql[:200]}...")
        print(f"Ground Truth SQL: {gt_sql[:200]}...")
        print(f"Error: {msg[:150]}...")

    # Summary for Table 5
    print("\n" + "=" * 80)
    print("SUMMARY FOR TABLE 5")
    print("=" * 80)
    print(f"1. SQL Syntax Errors: {len(syntax_errors)}/{total} ({len(syntax_errors)/total:.1%})")
    print(f"2. Record Mismatches: {len(record_mismatches)}/{total} ({len(record_mismatches)/total:.1%})")
    print(f"3. Missing WHERE Clauses: {len(missing_where)}/{total} ({len(missing_where)/total:.1%})")
    print(f"4. Incorrect WHERE Conditions: {len(incorrect_where)}/{total} ({len(incorrect_where)/total:.1%})")
    print(f"5. Missing/Incorrect SELECT Columns: {len(select_errors)}/{total} ({len(select_errors)/total:.1%})")
    print(f"6. Missing Comparison Operators: {len(missing_ops)}/{total} ({len(missing_ops)/total:.1%})")
    print("=" * 80)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    records_dir = os.path.join(base_dir, 'records')

    model_name = "t5_ft_final"
    gt_sql_path = os.path.join(data_dir, 'dev.sql')
    model_sql_path = os.path.join(results_dir, f'{model_name}_dev.sql')
    gt_record_path = os.path.join(records_dir, 'ground_truth_dev.pkl')
    model_record_path = os.path.join(records_dir, f'{model_name}_dev.pkl')
    nl_path = os.path.join(data_dir, 'dev.nl')

    analyze_errors(model_name, gt_sql_path, model_sql_path, gt_record_path, model_record_path, nl_path)

