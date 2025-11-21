#!/usr/bin/env python3
"""
Data Exploration Script for T5 Text-to-SQL
Examines the actual data files and provides insights for implementation
"""

import os

def explore_data_files():
    """Explore the structure of data files"""
    data_folder = 'data'
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder '{data_folder}' not found")
        return
    
    print("🔍 EXPLORING TEXT-TO-SQL DATA")
    print("=" * 50)
    
    # Check file sizes and sample content
    files_to_check = [
        'train.nl', 'train.sql', 
        'dev.nl', 'dev.sql',
        'test.nl',
        'flight_database.schema',
        'alignment.txt'
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n📁 {filename}")
            print(f"   Lines: {len(lines)}")
            print(f"   Sample (first 2 lines):")
            for i, line in enumerate(lines[:2]):
                print(f"   {i+1}: {line.strip()[:80]}{'...' if len(line.strip()) > 80 else ''}")
        else:
            print(f"\n❌ {filename} - NOT FOUND")
    
    # Check database file
    db_file = os.path.join(data_folder, 'flight_database.db')
    if os.path.exists(db_file):
        print(f"\n📊 flight_database.db")
        print(f"   Size: {os.path.getsize(db_file):,} bytes")
        print("   ✅ Database file found")
    else:
        print(f"\n❌ flight_database.db - NOT FOUND")

def analyze_schema_file():
    """Analyze the schema file structure"""
    schema_path = os.path.join('data', 'flight_database.schema')
    
    if not os.path.exists(schema_path):
        print("❌ Schema file not found")
        return
    
    print("\n📋 DATABASE SCHEMA ANALYSIS")
    print("=" * 50)
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_content = f.read()
    
    print("Schema file contents:")
    print(schema_content[:500] + "..." if len(schema_content) > 500 else schema_content)

def analyze_alignment_file():
    """Analyze the alignment file"""
    align_path = os.path.join('data', 'alignment.txt')
    
    if not os.path.exists(align_path):
        print("❌ Alignment file not found")
        return
    
    print("\n🔗 ALIGNMENT ANALYSIS")
    print("=" * 50)
    
    with open(align_path, 'r', encoding='utf-8') as f:
        align_lines = f.readlines()
    
    print(f"Alignment entries: {len(align_lines)}")
    print("Sample alignments:")
    for i, line in enumerate(align_lines[:5]):
        print(f"  {i+1}: {line.strip()}")

def suggest_data_processing_strategies():
    """Suggest data processing strategies based on available files"""
    print("\n💡 DATA PROCESSING STRATEGIES")
    print("=" * 50)
    
    strategies = [
        {
            "name": "Schema-Enhanced Input",
            "description": "Prepend relevant schema info to natural language queries",
            "implementation": "Add table/column info before NL query for T5 encoder",
            "files_used": ["flight_database.schema"]
        },
        {
            "name": "Alignment-Aware Processing",
            "description": "Use alignment.txt for better entity linking",
            "implementation": "Enhance entity recognition using provided alignments",
            "files_used": ["alignment.txt"]
        },
        {
            "name": "Task-Specific Prefixes",
            "description": "Optimize T5 prefix for SQL generation",
            "implementation": "Try 'Generate SQL query:', 'Convert to SQL:' instead of generic",
            "files_used": ["train.nl", "train.sql"]
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Description: {strategy['description']}")
        print(f"   Implementation: {strategy['implementation']}")
        print(f"   Files used: {', '.join(strategy['files_used'])}")

def main():
    print("🚀 T5 TEXT-TO-SQL DATA EXPLORATION")
    print("=" * 60)
    
    explore_data_files()
    analyze_schema_file()
    analyze_alignment_file()
    suggest_data_processing_strategies()
    
    print("\n" + "=" * 60)
    print("✅ DATA EXPLORATION COMPLETE")
    print("\nKey Insights:")
    print("1. Use schema file for enhanced input processing")
    print("2. Use alignment file for better entity linking")
    print("3. Standard T5 approach: 'translate English to SQL: ' + NL query")
    print("4. prompting.py is for alternative LLM approach (Gemma)")
    print("5. Stick with T5 fine-tuning for main assignment")

if __name__ == "__main__":
    main()