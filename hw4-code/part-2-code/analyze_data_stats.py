#!/usr/bin/env python3
"""
Data Statistics Analysis for Q4: T5 Text-to-SQL
Analyzes data before and after preprocessing using T5 tokenizer
"""

import os
from collections import Counter
from transformers import T5TokenizerFast
import nltk
from nltk.tokenize import word_tokenize
import statistics

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data_files(data_folder, split):
    """Load natural language and SQL files for a given split"""
    nl_file = os.path.join(data_folder, f"{split}.nl")
    sql_file = os.path.join(data_folder, f"{split}.sql")
    
    # Read natural language queries
    with open(nl_file, 'r', encoding='utf-8') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Read SQL queries (might not exist for test)
    sql_queries = []
    if os.path.exists(sql_file):
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_queries = [line.strip() for line in f.readlines()]
    
    return nl_queries, sql_queries

def analyze_before_preprocessing(nl_queries, sql_queries):
    """Analyze data statistics before preprocessing"""
    stats = {}
    
    # Number of examples
    stats['num_examples'] = len(nl_queries)
    
    # Mean sentence length (using word tokenization)
    nl_lengths = [len(word_tokenize(query)) for query in nl_queries]
    stats['mean_sentence_length'] = statistics.mean(nl_lengths)
    
    # Mean SQL query length
    if sql_queries:
        sql_lengths = [len(word_tokenize(query)) for query in sql_queries]
        stats['mean_sql_length'] = statistics.mean(sql_lengths)
    else:
        stats['mean_sql_length'] = None
    
    # Vocabulary sizes
    nl_vocab = set()
    sql_vocab = set()
    
    for query in nl_queries:
        nl_vocab.update(word_tokenize(query.lower()))
    
    if sql_queries:
        for query in sql_queries:
            sql_vocab.update(word_tokenize(query.lower()))
    
    stats['nl_vocab_size'] = len(nl_vocab)
    stats['sql_vocab_size'] = len(sql_vocab) if sql_queries else None
    
    return stats

def analyze_after_preprocessing(nl_queries, sql_queries, tokenizer):
    """Analyze data statistics after T5 tokenization preprocessing"""
    stats = {}
    
    # Number of examples (same as before preprocessing)
    stats['num_examples'] = len(nl_queries)
    
    # Mean sentence length (using T5 tokenizer)
    nl_lengths = []
    for query in nl_queries:
        # Add T5 task prefix
        prefixed_query = "translate English to SQL: " + query
        tokens = tokenizer(prefixed_query, truncation=True, max_length=512)['input_ids']
        nl_lengths.append(len(tokens))
    
    stats['mean_sentence_length'] = statistics.mean(nl_lengths)
    
    # Mean SQL query length (using T5 tokenizer)
    if sql_queries:
        sql_lengths = []
        for query in sql_queries:
            tokens = tokenizer(query, truncation=True, max_length=512)['input_ids']
            sql_lengths.append(len(tokens))
        stats['mean_sql_length'] = statistics.mean(sql_lengths)
    else:
        stats['mean_sql_length'] = None
    
    # Vocabulary sizes (T5 subword tokens)
    nl_vocab = set()
    sql_vocab = set()
    
    for query in nl_queries:
        prefixed_query = "translate English to SQL: " + query
        tokens = tokenizer(prefixed_query, truncation=True, max_length=512)['input_ids']
        nl_vocab.update(tokens)
    
    if sql_queries:
        for query in sql_queries:
            tokens = tokenizer(query, truncation=True, max_length=512)['input_ids']
            sql_vocab.update(tokens)
    
    stats['nl_vocab_size'] = len(nl_vocab)
    stats['sql_vocab_size'] = len(sql_vocab) if sql_queries else None
    
    return stats

def print_statistics_table(train_stats, dev_stats, title, before_preprocessing=True):
    """Print formatted statistics table"""
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Statistics Name':<30} {'Train':<10} {'Dev':<10}")
    print("-" * 60)
    
    print(f"{'Number of examples':<30} {train_stats['num_examples']:<10} {dev_stats['num_examples']:<10}")
    print(f"{'Mean sentence length':<30} {train_stats['mean_sentence_length']:<10.2f} {dev_stats['mean_sentence_length']:<10.2f}")
    
    if train_stats['mean_sql_length'] is not None:
        print(f"{'Mean SQL query length':<30} {train_stats['mean_sql_length']:<10.2f} {dev_stats['mean_sql_length']:<10.2f}")
    else:
        print(f"{'Mean SQL query length':<30} {'N/A':<10} {'N/A':<10}")
    
    print(f"{'Vocabulary size (NL)':<30} {train_stats['nl_vocab_size']:<10} {dev_stats['nl_vocab_size']:<10}")
    
    if train_stats['sql_vocab_size'] is not None:
        print(f"{'Vocabulary size (SQL)':<30} {train_stats['sql_vocab_size']:<10} {dev_stats['sql_vocab_size']:<10}")
    else:
        print(f"{'Vocabulary size (SQL)':<30} {'N/A':<10} {'N/A':<10}")

def main():
    try:
        data_folder = 'data'  # Adjust path as needed
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            print(f"Error: Data folder '{data_folder}' not found.")
            print("Please ensure you have the data directory with train.nl, train.sql, dev.nl, dev.sql files")
            return
        
        # Initialize T5 tokenizer
        print("Loading T5 tokenizer...")
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Load data
        print("Loading training data...")
        train_nl, train_sql = load_data_files(data_folder, 'train')
        
        print("Loading development data...")
        dev_nl, dev_sql = load_data_files(data_folder, 'dev')
        
        print(f"Loaded {len(train_nl)} training examples and {len(dev_nl)} dev examples")
        
        # Analyze before preprocessing
        print("\nAnalyzing statistics before preprocessing...")
        train_before = analyze_before_preprocessing(train_nl, train_sql)
        dev_before = analyze_before_preprocessing(dev_nl, dev_sql)
        
        # Analyze after preprocessing
        print("Analyzing statistics after T5 preprocessing...")
        train_after = analyze_after_preprocessing(train_nl, train_sql, tokenizer)
        dev_after = analyze_after_preprocessing(dev_nl, dev_sql, tokenizer)
        
        # Print results
        print_statistics_table(train_before, dev_before, 
                              "Table 1: Data statistics before preprocessing", 
                              before_preprocessing=True)
        
        print_statistics_table(train_after, dev_after, 
                              "Table 2: Data statistics after T5 preprocessing (google-t5/t5-small)", 
                              before_preprocessing=False)
        
        # Additional insights
        print(f"\n\nAdditional Analysis:")
        print(f"T5 tokenizer vocabulary size: {tokenizer.vocab_size:,}")
        print(f"T5 model name: google-t5/t5-small")
        
        # Save results to file for report
        with open('q4_data_statistics.txt', 'w') as f:
            f.write("Q4: Data Statistics and Processing Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Table 1: Data statistics before any pre-processing\n")
            f.write(f"Number of examples: Train={train_before['num_examples']}, Dev={dev_before['num_examples']}\n")
            f.write(f"Mean sentence length: Train={train_before['mean_sentence_length']:.2f}, Dev={dev_before['mean_sentence_length']:.2f}\n")
            if train_before['mean_sql_length']:
                f.write(f"Mean SQL query length: Train={train_before['mean_sql_length']:.2f}, Dev={dev_before['mean_sql_length']:.2f}\n")
            f.write(f"Vocabulary size (natural language): Train={train_before['nl_vocab_size']}, Dev={dev_before['nl_vocab_size']}\n")
            if train_before['sql_vocab_size']:
                f.write(f"Vocabulary size (SQL): Train={train_before['sql_vocab_size']}, Dev={dev_before['sql_vocab_size']}\n")
            
            f.write("\nTable 2: Data statistics after pre-processing (T5 tokenization)\n")
            f.write("Model name: google-t5/t5-small\n")
            f.write(f"Mean sentence length: Train={train_after['mean_sentence_length']:.2f}, Dev={dev_after['mean_sentence_length']:.2f}\n")
            if train_after['mean_sql_length']:
                f.write(f"Mean SQL query length: Train={train_after['mean_sql_length']:.2f}, Dev={dev_after['mean_sql_length']:.2f}\n")
            f.write(f"Vocabulary size (natural language): Train={train_after['nl_vocab_size']}, Dev={dev_after['nl_vocab_size']}\n")
            if train_after['sql_vocab_size']:
                f.write(f"Vocabulary size (SQL): Train={train_after['sql_vocab_size']}, Dev={dev_after['sql_vocab_size']}\n")
        
        print(f"\nResults saved to: q4_data_statistics.txt")
        print("✅ Data analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in data analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()