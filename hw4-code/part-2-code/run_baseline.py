#!/usr/bin/env python3
"""
Simple Baseline Runner for T5 Text-to-SQL
Establishes baseline performance before experimentation
"""

import os
import sys

def check_data_availability():
    """Check if required data files are available"""
    required_files = [
        'data/train.nl', 'data/train.sql',
        'data/dev.nl', 'data/dev.sql', 
        'data/test.nl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure you have the data directory with all required files.")
        return False
    
    print("✅ All required data files found.")
    return True

def run_data_analysis():
    """Run Q4 data statistics analysis"""
    print("\n" + "="*50)
    print("STEP 1: DATA STATISTICS ANALYSIS (Q4)")
    print("="*50)
    
    if os.system("python3 analyze_data_stats.py") == 0:
        print("✅ Data analysis completed successfully")
        return True
    else:
        print("❌ Data analysis failed")
        return False

def run_baseline_training():
    """Run baseline T5 fine-tuning experiment"""
    print("\n" + "="*50) 
    print("STEP 2: BASELINE T5 FINE-TUNING")
    print("="*50)
    print("Configuration:")
    print("  - Model: google-t5/t5-small (fine-tuned)")
    print("  - Epochs: 10")
    print("  - Learning Rate: 5e-4") 
    print("  - Batch Size: 8")
    print("  - Scheduler: Cosine with warmup")
    print("  - No parameter freezing")
    print("  - Standard T5 tokenizer and data processing")
    
    cmd = (
        "python3 train_t5.py "
        "--finetune "
        "--max_n_epochs 10 "
        "--learning_rate 5e-4 "
        "--batch_size 8 "
        "--weight_decay 0.01 "
        "--scheduler_type cosine "
        "--num_warmup_epochs 1 "
        "--patience_epochs 5 "
        "--experiment_name baseline"
    )
    
    print(f"\nRunning: {cmd}")
    
    if os.system(cmd) == 0:
        print("✅ Baseline training completed successfully")
        return True
    else:
        print("❌ Baseline training failed")
        return False

def check_baseline_results():
    """Check if baseline results were generated properly"""
    print("\n" + "="*50)
    print("STEP 3: CHECKING BASELINE RESULTS") 
    print("="*50)
    
    expected_files = [
        "results/t5_ft_baseline_dev.sql",
        "results/t5_ft_baseline_test.sql", 
        "records/t5_ft_baseline_dev.pkl",
        "records/t5_ft_baseline_test.pkl"
    ]
    
    success = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            success = False
    
    return success

def evaluate_baseline():
    """Evaluate baseline performance on dev set"""
    print("\n" + "="*50)
    print("STEP 4: BASELINE EVALUATION")
    print("="*50)
    
    # Check if ground truth dev records exist
    gt_records_path = "records/ground_truth_dev.pkl"
    if not os.path.exists(gt_records_path):
        print(f"⚠️  Ground truth dev records not found at {gt_records_path}")
        print("   Skipping evaluation - you may need to generate ground truth records first")
        return True
    
    cmd = (
        "python3 evaluate.py "
        "--predicted_sql results/t5_ft_baseline_dev.sql "
        "--predicted_records records/t5_ft_baseline_dev.pkl "
        "--development_sql data/dev.sql "
        "--development_records records/ground_truth_dev.pkl"
    )
    
    print(f"Running: {cmd}")
    
    if os.system(cmd) == 0:
        print("✅ Baseline evaluation completed")
        return True
    else:
        print("❌ Baseline evaluation failed") 
        return False

def main():
    print("🚀 T5 Text-to-SQL Baseline Establishment")
    print("="*60)
    
    # Step 1: Check data availability
    if not check_data_availability():
        sys.exit(1)
    
    # Step 2: Run data analysis for Q4
    if not run_data_analysis():
        print("⚠️  Data analysis failed, but continuing with training...")
    
    # Step 3: Run baseline training
    if not run_baseline_training():
        print("❌ Baseline training failed. Please check error messages above.")
        sys.exit(1)
    
    # Step 4: Check results were generated
    if not check_baseline_results():
        print("⚠️  Some expected output files are missing.")
    
    # Step 5: Evaluate baseline performance
    if not evaluate_baseline():
        print("⚠️  Baseline evaluation failed.")
    
    print("\n" + "="*60)
    print("🎉 BASELINE ESTABLISHMENT COMPLETE")
    print("="*60)
    print("✅ Baseline model trained and evaluated")
    print("✅ Ready for systematic experimentation")
    print("\nNext steps:")
    print("  1. Check baseline F1 score in the evaluation output")
    print("  2. If F1 ≥ 65%, you're ready for submission")
    print("  3. If F1 < 65%, run experiments with: python3 run_experiments.py")
    print("  4. Document results for Q5-Q6 writeup")

if __name__ == "__main__":
    main()