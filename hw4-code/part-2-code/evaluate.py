from argparse import ArgumentParser
from utils import compute_metrics

parser = ArgumentParser()
parser.add_argument("-ps", "--predicted_sql", dest = "pred_sql",
    required = True, help = "path to your model's predicted SQL queries")
parser.add_argument("-pr", "--predicted_records", dest = "pred_records",
    required = True, help = "path to the predicted development database records")
parser.add_argument("-ds", "--development_sql", dest = "dev_sql",
    required = True, help = "path to the ground-truth development SQL queries")
parser.add_argument("-dr", "--development_records", dest = "dev_records",
    required = True, help = "path to the ground-truth development database records")

args = parser.parse_args()
sql_em, record_em, record_f1, error_msgs = compute_metrics(args.dev_sql, args.pred_sql, args.dev_records, args.pred_records)

# Calculate SQL Error Rate
num_errors = sum(1 for msg in error_msgs if msg != "")
total_queries = len(error_msgs)
error_rate = num_errors / total_queries if total_queries > 0 else 0

print("=" * 80)
print("Evaluation Metrics")
print("=" * 80)
print(f"Query EM:        {sql_em*100:.2f}% ({sql_em:.4f})")
print(f"Record EM:       {record_em*100:.2f}% ({record_em:.4f})")
print(f"Record F1:       {record_f1*100:.2f}% ({record_f1:.4f})")
print(f"SQL Error Rate:  {error_rate*100:.2f}% ({num_errors}/{total_queries})")
print("=" * 80)