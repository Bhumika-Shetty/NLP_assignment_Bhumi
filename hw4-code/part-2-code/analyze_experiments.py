#!/usr/bin/env python3
"""
Experimental Results Analysis Tool
==================================

Analyzes the results from the experimental pipeline to help identify
the best configuration for final training.
"""

import os
import re
import glob
from collections import defaultdict

def extract_best_f1(experiment_dir):
    """Extract the best F1 score from experiment logs"""
    # Look for saved model files or result files
    try:
        # Check if experiment completed (has saved model)
        best_model_path = os.path.join(experiment_dir, "best_model.pt")
        if not os.path.exists(best_model_path):
            return None, "Experiment incomplete"
        
        # Try to find F1 score from results files
        results_pattern = f"results/t5_ft_{os.path.basename(experiment_dir)}_dev.sql"
        if os.path.exists(results_pattern):
            # This would require parsing evaluation output
            return "Check manually", "Found results file"
        
        return "Unknown", "Completed but F1 not extracted"
    
    except Exception as e:
        return None, f"Error: {e}"

def analyze_experiments():
    """Analyze all experimental results"""
    print("🔬 T5 EXPERIMENTAL RESULTS ANALYSIS")
    print("="*60)
    
    # Find all experiment directories
    experiment_dirs = glob.glob("checkpoints/ft_experiments/*")
    
    if not experiment_dirs:
        print("❌ No experiment directories found!")
        print("   Make sure experiments have been run and saved to checkpoints/")
        return
    
    print(f"📁 Found {len(experiment_dirs)} experiments:")
    
    results = []
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        f1_score, status = extract_best_f1(exp_dir)
        
        results.append({
            'name': exp_name,
            'f1': f1_score,
            'status': status,
            'path': exp_dir
        })
        
        print(f"   📊 {exp_name}: {status}")
    
    # Group by experiment type
    experiment_groups = {
        'Preprocessing': [r for r in results if 'preprocess' in r['name'] or 'baseline' in r['name']],
        'Architecture': [r for r in results if 'arch' in r['name']],
        'Generation': [r for r in results if 'gen' in r['name']],
        'Learning Rate': [r for r in results if 'lr' in r['name']],
        'Other': [r for r in results if not any(x in r['name'] for x in ['preprocess', 'baseline', 'arch', 'gen', 'lr'])]
    }
    
    print(f"\n📈 EXPERIMENT SUMMARY BY CATEGORY:")
    print("-" * 60)
    
    for category, exps in experiment_groups.items():
        if exps:
            print(f"\n🔬 {category} Experiments:")
            for exp in exps:
                status_icon = "✅" if "Completed" in exp['status'] or exp['f1'] else "⚠️"
                print(f"   {status_icon} {exp['name']}: {exp['status']}")
    
    print(f"\n💡 MANUAL ANALYSIS REQUIRED:")
    print("To get F1 scores, check the terminal output or result files:")
    print("1. Look at training logs for 'NEW BEST F1' messages")
    print("2. Check results/ directory for generated SQL files")
    print("3. Run evaluation script on dev sets")
    
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Identify experiments with highest F1 scores")
    print("2. Look for low error rates and good SQL generation")
    print("3. Choose best combination of:")
    print("   • Learning rate (target: minimize error rate)")
    print("   • Generation parameters (balance F1 vs speed)")
    print("   • Architectural choices (if any improve performance)")
    
    return results

def suggest_final_config():
    """Suggest final configuration based on typical results"""
    print(f"\n🏆 TYPICAL BEST CONFIGURATION SUGGESTIONS:")
    print("-" * 60)
    
    print(f"Based on common T5 text-to-SQL results:")
    print(f"")
    print(f"🎯 Conservative (Safe Choice):")
    print(f"   • Learning rate: 3e-4 or 5e-4")
    print(f"   • Generation: num_beams=3-5, max_length=256-512") 
    print(f"   • Architecture: Full model (no freezing)")
    print(f"   • Weight decay: 0.01")
    print(f"")
    print(f"🚀 Aggressive (High Performance):")
    print(f"   • Learning rate: 5e-4 to 8e-4")
    print(f"   • Generation: num_beams=5-10, max_length=256")
    print(f"   • Architecture: Consider encoder freezing if overfitting")
    print(f"   • Weight decay: 0.01-0.1")
    print(f"")
    print(f"⚡ Speed Optimized:")
    print(f"   • Learning rate: 5e-4")
    print(f"   • Generation: num_beams=1 (greedy), max_length=128")
    print(f"   • Architecture: Full model")

if __name__ == "__main__":
    results = analyze_experiments()
    suggest_final_config()
    
    print(f"\n" + "="*60)
    print(f"📝 To run final training with best config:")
    print(f"python3 train_t5.py \\")
    print(f"  --max_n_epochs 30 \\")
    print(f"  --learning_rate [BEST] \\")
    print(f"  --num_beams [BEST] \\")
    print(f"  --max_gen_length [BEST] \\")
    print(f"  --experiment_name final_production")