#!/usr/bin/env python3
"""
T5 Text-to-SQL Experiment Runner and Baseline Establishment
Provides systematic experimentation framework for Part 2
"""

import os
import subprocess
import json
from datetime import datetime
import argparse

class ExperimentTracker:
    def __init__(self, results_dir="experiment_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def log_experiment(self, config, results, experiment_name):
        """Log experiment configuration and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "config": config,
            "results": results
        }
        
        log_file = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        print(f"Experiment logged to: {log_file}")
        return log_file

def run_baseline_experiment():
    """Run baseline experiment with simple hyperparameters"""
    print("=" * 60)
    print("RUNNING BASELINE EXPERIMENT")
    print("=" * 60)
    
    baseline_config = {
        "experiment_type": "baseline",
        "model": "google-t5/t5-small",
        "finetune": True,
        "max_n_epochs": 10,
        "learning_rate": 5e-4,
        "batch_size": 8,
        "weight_decay": 0.01,
        "scheduler_type": "cosine",
        "num_warmup_epochs": 1,
        "patience_epochs": 5,
        "frozen_parameters": "none",
        "data_processing": "standard"
    }
    
    cmd = [
        "python3", "train_t5.py",
        "--finetune",
        "--max_n_epochs", "10", 
        "--learning_rate", "5e-4",
        "--batch_size", "8",
        "--weight_decay", "0.01",
        "--scheduler_type", "cosine",
        "--num_warmup_epochs", "1",
        "--patience_epochs", "5",
        "--experiment_name", "baseline"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return baseline_config, {
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def run_parameter_freezing_experiments():
    """Run experiments with different parameter freezing strategies"""
    experiments = [
        {
            "name": "freeze_encoder",
            "description": "Freeze encoder, train decoder only",
            "args": ["--freeze_encoder"],
            "config": {"frozen_parameters": "encoder"}
        },
        {
            "name": "freeze_decoder",
            "description": "Freeze decoder, train encoder only", 
            "args": ["--freeze_decoder"],
            "config": {"frozen_parameters": "decoder"}
        },
        {
            "name": "freeze_embeddings",
            "description": "Freeze embeddings, train transformer layers",
            "args": ["--freeze_embeddings"],
            "config": {"frozen_parameters": "embeddings"}
        }
    ]
    
    results = []
    for exp in experiments:
        print("=" * 60)
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print(f"Description: {exp['description']}")
        print("=" * 60)
        
        base_cmd = [
            "python3", "train_t5.py",
            "--finetune",
            "--max_n_epochs", "5",
            "--learning_rate", "1e-4", 
            "--batch_size", "8",
            "--experiment_name", exp['name']
        ]
        
        cmd = base_cmd + exp['args']
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        config = {
            "experiment_type": "parameter_freezing",
            "model": "google-t5/t5-small",
            "finetune": True,
            "max_n_epochs": 5,
            "learning_rate": 1e-4,
            "batch_size": 8,
            **exp['config']
        }
        
        results.append({
            "experiment": exp['name'],
            "config": config,
            "results": {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        })
    
    return results

def run_data_processing_experiments():
    """Run experiments with different data processing strategies"""
    experiments = [
        {
            "name": "schema_enhanced",
            "description": "Include database schema in T5 input for better SQL generation",
            "implementation": "Prepend schema info to natural language queries"
        },
        {
            "name": "alignment_aware", 
            "description": "Use alignment.txt to improve entity linking",
            "implementation": "Enhance entity recognition using provided alignments"
        },
        {
            "name": "prefix_optimized",
            "description": "Optimize T5 task prefix for SQL domain",
            "implementation": "Try 'Generate SQL: ', 'Convert to SQL: ' instead of generic prefix"
        }
    ]
    
    results = []
    for exp in experiments:
        print("=" * 60)
        print(f"RUNNING DATA PROCESSING EXPERIMENT: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Implementation: {exp['implementation']}")
        print("=" * 60)
        
        base_cmd = [
            "python3", "train_t5.py",
            "--finetune",
            "--max_n_epochs", "8",
            "--learning_rate", "5e-4", 
            "--batch_size", "8",
            "--experiment_name", exp['name']
        ]
        
        # Add specific configurations for each data processing strategy
        if exp['name'] == 'schema_enhanced':
            cmd = base_cmd + ["--learning_rate", "3e-4"]  # Lower LR for richer input
        elif exp['name'] == 'alignment_aware':
            cmd = base_cmd + ["--weight_decay", "0.005"]  # Regularization for enhanced data
        elif exp['name'] == 'prefix_optimized':
            cmd = base_cmd + ["--max_n_epochs", "6"]  # Fewer epochs might be enough
        else:
            cmd = base_cmd
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        config = {
            "experiment_type": "data_processing",
            "model": "google-t5/t5-small", 
            "strategy": exp['name'],
            "description": exp['description'],
            "implementation": exp['implementation']
        }
        
        results.append({
            "experiment": exp['name'],
            "config": config,
            "results": {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        })
    
    return results

def run_hyperparameter_experiments():
    """Run experiments with different hyperparameters"""
    experiments = [
        {
            "name": "lr_5e5",
            "description": "Lower learning rate",
            "learning_rate": "5e-5"
        },
        {
            "name": "lr_3e4", 
            "description": "Higher learning rate",
            "learning_rate": "3e-4"
        },
        {
            "name": "batch_16",
            "description": "Larger batch size",
            "batch_size": "16"
        },
        {
            "name": "epochs_10",
            "description": "More training epochs",
            "max_n_epochs": "10"
        }
    ]
    
    results = []
    for exp in experiments:
        print("=" * 60) 
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print(f"Description: {exp['description']}")
        print("=" * 60)
        
        # Base configuration
        cmd = [
            "python3", "train_t5.py",
            "--finetune",
            "--max_n_epochs", exp.get("max_n_epochs", "5"),
            "--learning_rate", exp.get("learning_rate", "1e-4"),
            "--batch_size", exp.get("batch_size", "8"),
            "--experiment_name", exp['name']
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        config = {
            "experiment_type": "hyperparameter_tuning", 
            "model": "google-t5/t5-small",
            "finetune": True,
            "max_n_epochs": int(exp.get("max_n_epochs", "5")),
            "learning_rate": float(exp.get("learning_rate", "1e-4")),
            "batch_size": int(exp.get("batch_size", "8"))
        }
        
        results.append({
            "experiment": exp['name'],
            "config": config,
            "results": {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        })
    
    return results

def generate_experiment_report(tracker):
    """Generate a comprehensive experiment report"""
    print("\n" + "=" * 60)
    print("GENERATING EXPERIMENT REPORT")
    print("=" * 60)
    
    # Read all experiment logs
    log_files = [f for f in os.listdir(tracker.results_dir) if f.endswith('.json')]
    
    if not log_files:
        print("No experiment logs found.")
        return
    
    experiments = []
    for log_file in log_files:
        with open(os.path.join(tracker.results_dir, log_file), 'r') as f:
            experiments.append(json.load(f))
    
    # Generate markdown report
    report_file = os.path.join(tracker.results_dir, "experiment_report.md")
    with open(report_file, 'w') as f:
        f.write("# T5 Text-to-SQL Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Summary\n\n")
        f.write(f"Total experiments run: {len(experiments)}\n\n")
        
        for exp in experiments:
            f.write(f"### {exp['experiment_name']}\n")
            f.write(f"**Type**: {exp['config'].get('experiment_type', 'unknown')}\n")
            f.write(f"**Timestamp**: {exp['timestamp']}\n")
            f.write("**Configuration**:\n")
            for key, value in exp['config'].items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")
            
            if exp['results']['return_code'] == 0:
                f.write("**Status**: ✅ Success\n")
            else:
                f.write("**Status**: ❌ Failed\n")
                f.write(f"**Error**: {exp['results']['stderr'][:200]}...\n")
            f.write("\n")
    
    print(f"Experiment report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="T5 Text-to-SQL Experiment Runner")
    parser.add_argument("--baseline", action="store_true", help="Run baseline experiment only")
    parser.add_argument("--freezing", action="store_true", help="Run parameter freezing experiments")
    parser.add_argument("--hyperparams", action="store_true", help="Run hyperparameter experiments")
    parser.add_argument("--dataproc", action="store_true", help="Run data processing experiments") 
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--report", action="store_true", help="Generate experiment report only")
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker()
    
    if args.report:
        generate_experiment_report(tracker)
        return
    
    if args.baseline or args.all:
        config, results = run_baseline_experiment()
        tracker.log_experiment(config, results, "baseline")
    
    if args.freezing or args.all:
        freeze_results = run_parameter_freezing_experiments()
        for result in freeze_results:
            tracker.log_experiment(result['config'], result['results'], result['experiment'])
    
    if args.hyperparams or args.all:
        hyperparam_results = run_hyperparameter_experiments()
        for result in hyperparam_results:
            tracker.log_experiment(result['config'], result['results'], result['experiment'])
    
    if args.dataproc or args.all:
        dataproc_results = run_data_processing_experiments()
        for result in dataproc_results:
            tracker.log_experiment(result['config'], result['results'], result['experiment'])
    
    # Generate final report
    generate_experiment_report(tracker)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("Check experiment_results/ directory for detailed logs and reports")

if __name__ == "__main__":
    main()