#!/usr/bin/env python3
"""
T5 Text-to-SQL Experimental Pipeline
====================================

Systematic experiments to optimize T5 performance:
1. Data Preprocessing Experiments
2. Architectural Experiments  
3. Generation Parameter Experiments
4. Learning Rate Experiments
5. Final Best Configuration Training

Each experiment runs 5-7 epochs to quickly assess effectiveness.
"""

import os
import subprocess
import time
from datetime import datetime

def run_experiment(command, experiment_name, description):
    """Run a single experiment and log results"""
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"📝 Description: {description}")
    print(f"⚙️  Command: {command}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)
    
    start_time = time.time()
    try:
        # Use timeout and better process handling
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=False,
            timeout=3600  # 1 hour timeout per experiment
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {experiment_name} completed successfully in {duration/60:.1f} minutes")
            print(f"⚡ Moving to next experiment...")
            time.sleep(2)  # Brief pause before next experiment
        else:
            print(f"\n❌ {experiment_name} failed with return code {result.returncode}")
            print(f"🔄 Continuing with remaining experiments...")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {experiment_name} timed out after 1 hour - skipping")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 {experiment_name} interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ {experiment_name} crashed: {e}")
        print(f"🔄 Continuing with remaining experiments...")
        return False

def main():
    print("🚀 T5 TEXT-TO-SQL EXPERIMENTAL PIPELINE")
    print("="*80)
    print("📊 Running systematic experiments to optimize model performance")
    print("⏱️  Each experiment: 5-7 epochs for quick assessment")
    print("🎯 Goal: Find best configuration for final 25-30 epoch training")
    print("="*80)
    
    # Base command template
    base_cmd = "python3 train_t5.py --max_n_epochs 6 --skip_test_inference"
    
    successful_experiments = []
    failed_experiments = []
    
    # ========================================
    # 1. DATA PREPROCESSING EXPERIMENTS
    # ========================================
    print(f"\n🔬 PHASE 1: DATA PREPROCESSING EXPERIMENTS")
    
    preprocessing_experiments = [
        {
            'name': 'shorter_sequences',
            'cmd': f"{base_cmd} --max_gen_length 256 --experiment_name preprocess_short_seq",
            'desc': 'Shorter max generation length (256 vs 512)'
        },
        {
            'name': 'longer_sequences',  
            'cmd': f"{base_cmd} --max_gen_length 768 --experiment_name preprocess_long_seq",
            'desc': 'Longer max generation length (768 vs 512)'
        }
    ]
    
    for exp in preprocessing_experiments:
        success = run_experiment(exp['cmd'], exp['name'], exp['desc'])
        if success:
            successful_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
    
    # ========================================
    # 2. ARCHITECTURAL EXPERIMENTS
    # ========================================
    print(f"\n🏗️  PHASE 2: ARCHITECTURAL EXPERIMENTS")
    
    # Note: These require modifications to t5_utils.py
    print("📋 Creating architectural variants...")
    
    architectural_experiments = [
        {
            'name': 'freeze_encoder',
            'cmd': f"{base_cmd} --experiment_name arch_freeze_encoder",
            'desc': 'Freeze encoder, train only decoder',
            'modification': 'freeze_encoder'
        },
        {
            'name': 'smaller_lr_full_model',
            'cmd': f"{base_cmd} --learning_rate 1e-4 --experiment_name arch_small_lr_full",
            'desc': 'Full model with much smaller learning rate'
        },
        {
            'name': 'higher_weight_decay',
            'cmd': f"{base_cmd} --weight_decay 0.1 --experiment_name arch_high_decay",
            'desc': 'Higher weight decay for regularization'
        },
        {
            'name': 'no_weight_decay',
            'cmd': f"{base_cmd} --weight_decay 0.0 --experiment_name arch_no_decay", 
            'desc': 'No weight decay'
        }
    ]
    
    for exp in architectural_experiments:
        if exp.get('modification') == 'freeze_encoder':
            print(f"⚠️  Skipping {exp['name']} - requires manual code modification")
            print(f"   To enable: Uncomment encoder freezing in t5_utils.py")
            continue
            
        success = run_experiment(exp['cmd'], exp['name'], exp['desc'])
        if success:
            successful_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
    
    # ========================================
    # 3. GENERATION PARAMETER EXPERIMENTS
    # ========================================
    print(f"\n🎲 PHASE 3: GENERATION PARAMETER EXPERIMENTS")
    
    generation_experiments = [
        {
            'name': 'greedy_decoding',
            'cmd': f"{base_cmd} --num_beams 1 --experiment_name gen_greedy",
            'desc': 'Greedy decoding (num_beams=1)'
        },
        {
            'name': 'beam_search_3',
            'cmd': f"{base_cmd} --num_beams 3 --experiment_name gen_beam3",
            'desc': 'Small beam search (num_beams=3)'
        },
        {
            'name': 'beam_search_10',
            'cmd': f"{base_cmd} --num_beams 10 --experiment_name gen_beam10",
            'desc': 'Large beam search (num_beams=10)'
        },
        {
            'name': 'beam_search_short',
            'cmd': f"{base_cmd} --num_beams 5 --max_gen_length 128 --experiment_name gen_beam5_short",
            'desc': 'Current beams with shorter sequences'
        }
    ]
    
    for exp in generation_experiments:
        success = run_experiment(exp['cmd'], exp['name'], exp['desc'])
        if success:
            successful_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
    
    # ========================================
    # 4. LEARNING RATE EXPERIMENTS
    # ========================================
    print(f"\n📈 PHASE 4: LEARNING RATE EXPERIMENTS")
    
    lr_experiments = [
        {
            'name': 'lr_very_low',
            'cmd': f"{base_cmd} --learning_rate 1e-4 --experiment_name lr_1e4",
            'desc': 'Very low learning rate (1e-4)'
        },
        {
            'name': 'lr_low',
            'cmd': f"{base_cmd} --learning_rate 3e-4 --experiment_name lr_3e4", 
            'desc': 'Low learning rate (3e-4)'
        },
        {
            'name': 'lr_current',
            'cmd': f"{base_cmd} --learning_rate 5e-4 --experiment_name lr_5e4_baseline",
            'desc': 'Current learning rate (5e-4) - baseline'
        },
        {
            'name': 'lr_high',
            'cmd': f"{base_cmd} --learning_rate 8e-4 --experiment_name lr_8e4",
            'desc': 'Higher learning rate (8e-4)'
        },
        {
            'name': 'lr_very_high',
            'cmd': f"{base_cmd} --learning_rate 1e-3 --experiment_name lr_1e3",
            'desc': 'High learning rate (1e-3)'
        }
    ]
    
    for exp in lr_experiments:
        success = run_experiment(exp['cmd'], exp['name'], exp['desc'])
        if success:
            successful_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
    
    # ========================================
    # 5. EXPERIMENT SUMMARY & RECOMMENDATIONS
    # ========================================
    print(f"\n📊 EXPERIMENTAL PIPELINE COMPLETE!")
    print("="*80)
    print(f"✅ Successful experiments ({len(successful_experiments)}):")
    for exp in successful_experiments:
        print(f"   • {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments ({len(failed_experiments)}):")
        for exp in failed_experiments:
            print(f"   • {exp}")
    
    print(f"\n🔍 NEXT STEPS:")
    print(f"1. 📈 Check results in each experiment directory")
    print(f"2. 📊 Compare F1 scores across experiments") 
    print(f"3. 🎯 Identify best configuration")
    print(f"4. 🚀 Run final training with best config for 25-30 epochs")
    
    print(f"\n💡 RECOMMENDED ANALYSIS COMMANDS:")
    print(f"# Check experiment results")
    print(f"ls -la checkpoints/*/")
    print(f"grep 'Best F1' results/*.sql")
    
    print(f"\n🏆 FINAL TRAINING COMMAND TEMPLATE:")
    print(f"python3 train_t5.py \\")
    print(f"  --max_n_epochs 30 \\")
    print(f"  --learning_rate [BEST_LR] \\")
    print(f"  --num_beams [BEST_BEAMS] \\")
    print(f"  --max_gen_length [BEST_LENGTH] \\")
    print(f"  --experiment_name final_best_config")

if __name__ == "__main__":
    main()