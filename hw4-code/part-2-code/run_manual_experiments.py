#!/usr/bin/env python3
"""
Manual Experiment Runner with Logging
=====================================

Run experiments manually with automatic logging and results tracking.
"""

import os
import subprocess
import time
import signal
import re
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.main_log = os.path.join(log_dir, "experiments_summary.log")
        
        # Initialize main log
        with open(self.main_log, "w") as f:
            f.write(f"T5 EXPERIMENTAL LOG - Started {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def run_experiment(self, command, exp_name, description):
        """Run experiment with comprehensive logging and auto-kill on completion"""
        print(f"\n🧪 STARTING: {exp_name}")
        print(f"📝 {description}")
        print(f"⚙️  Command: {command}")
        
        # Create individual log file
        exp_log = os.path.join(self.log_dir, f"{exp_name}.log")
        
        start_time = time.time()
        start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log to main summary
        with open(self.main_log, "a") as f:
            f.write(f"EXPERIMENT: {exp_name}\n")
            f.write(f"Started: {start_str}\n")
            f.write(f"Command: {command}\n")
            f.write(f"Description: {description}\n")
            f.write("-" * 40 + "\n")
        
        try:
            # Run experiment with process monitoring
            with open(exp_log, "w") as log_file:
                log_file.write(f"EXPERIMENT: {exp_name}\n")
                log_file.write(f"Started: {start_str}\n")
                log_file.write(f"Command: {command}\n")
                log_file.write("="*60 + "\n\n")
                log_file.flush()
                
                print(f"📊 Running... (logs: {exp_log})")
                
                # Start process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                # Monitor process and log file
                last_size = 0
                no_change_count = 0
                completion_detected = False
                
                while process.poll() is None:
                    time.sleep(30)  # Check every 30 seconds
                    
                    # Check if log file indicates completion
                    if os.path.exists(exp_log):
                        with open(exp_log, 'r') as f:
                            content = f.read()
                        
                        # Look for completion indicators
                        completion_phrases = [
                            "✅ TRAINING COMPLETE!",
                            "⚡ Experiment complete!",
                            "🎯 Best dev F1:",
                            "TRAINING COMPLETE",
                            "Experiment complete",
                            "Skipped test inference for speed"
                        ]
                        
                        if any(phrase in content for phrase in completion_phrases):
                            print(f"🎯 Completion detected in log - waiting 30s for cleanup...")
                            time.sleep(30)  # Give time for cleanup
                            completion_detected = True
                            break
                        
                        # Detect if log stopped growing (potential hang)
                        current_size = len(content)
                        if current_size == last_size:
                            no_change_count += 1
                            if no_change_count > 4:  # 2 minutes no change
                                print(f"⚠️  Log file stopped growing - checking for completion...")
                                if any(phrase in content for phrase in completion_phrases):
                                    completion_detected = True
                                    break
                        else:
                            no_change_count = 0
                            last_size = current_size
                
                # Force kill if completion detected
                if completion_detected and process.poll() is None:
                    print(f"🛑 Force killing completed experiment to prevent hang...")
                    os.killpg(os.getpgid(process.pid), 9)  # Kill process group
                    time.sleep(5)
                
                # Wait for process with timeout
                try:
                    result_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"🛑 Process timeout - force killing...")
                    os.killpg(os.getpgid(process.pid), 9)
                    result_code = -1
            
            duration = time.time() - start_time
            end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract F1 score from log
            f1_score = self.extract_f1_score(exp_log)
            
            success = completion_detected or result_code == 0
            status = "SUCCESS" if success else f"FAILED (code: {result_code})"
            
            print(f"✅ {status} - {duration/60:.1f} min")
            if f1_score:
                print(f"🎯 Best F1: {f1_score}")
            
            # Clean up any remaining processes
            self.cleanup_processes(exp_name)
            
            # Log results to main summary
            with open(self.main_log, "a") as f:
                f.write(f"Status: {status}\n")
                f.write(f"Duration: {duration/60:.1f} minutes\n")
                f.write(f"Ended: {end_str}\n")
                if f1_score:
                    f.write(f"Best F1: {f1_score}\n")
                f.write(f"Log file: {exp_log}\n")
                f.write("\n" + "="*60 + "\n\n")
            
            return success, f1_score
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ ERROR: {e}")
            
            # Clean up on error
            self.cleanup_processes(exp_name)
            
            with open(self.main_log, "a") as f:
                f.write(f"Status: ERROR - {e}\n")
                f.write(f"Duration: {duration/60:.1f} minutes\n")
                f.write(f"Log file: {exp_log}\n")
                f.write("\n" + "="*60 + "\n\n")
            
            return False, None
    
    def cleanup_processes(self, exp_name):
        """Clean up any hanging processes related to the experiment"""
        try:
            # Kill any remaining train_t5.py processes
            subprocess.run(['pkill', '-f', 'train_t5.py'], capture_output=True)
            subprocess.run(['pkill', '-f', exp_name], capture_output=True)
            time.sleep(2)
            print(f"🧹 Cleaned up processes for {exp_name}")
        except:
            pass
    
    def extract_f1_score(self, log_file):
        """Extract best F1 score from experiment log"""
        try:
            with open(log_file, "r") as f:
                content = f.read()
            
            # Look for F1 score patterns - including the specific one you mentioned
            import re
            patterns = [
                r"🏆 NEW BEST F1: (\d+\.?\d*)%",
                r"Best F1 score: (\d+\.?\d*)%", 
                r"Record F1:\s+(\d+\.?\d*)%",
                r"🎯 Best dev F1: (\d+\.?\d*)%",  # Your specific pattern
                r"Best dev F1: (\d+\.?\d*)%"
            ]
            
            scores = []
            for pattern in patterns:
                matches = re.findall(pattern, content)
                scores.extend([float(m) for m in matches])
            
            return f"{max(scores):.1f}%" if scores else None
            
        except:
            return None
    
    def print_summary(self):
        """Print experiment summary"""
        print(f"\n📊 EXPERIMENT SUMMARY")
        print("="*50)
        
        if os.path.exists(self.main_log):
            with open(self.main_log, "r") as f:
                content = f.read()
            print(content)
        else:
            print("No experiments logged yet.")

def main():
    logger = ExperimentLogger()
    
    print("🚀 T5 MANUAL EXPERIMENT RUNNER")
    print("="*50)
    print("📁 Logs will be saved to: experiment_logs/")
    print("📊 Main log: experiment_logs/experiments_summary.log")
    print()
    
    # Define experiments
    experiments = [
        {
            'name': 'preprocess_long_seq',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --max_gen_length 768 --experiment_name preprocess_long_seq',
            'desc': 'Longer max generation length (768 vs 512)'
        },
        {
            'name': 'arch_high_decay',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --weight_decay 0.1 --experiment_name arch_high_decay',
            'desc': 'Higher weight decay for regularization'
        },
        {
            'name': 'arch_no_decay',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --weight_decay 0.0 --experiment_name arch_no_decay',
            'desc': 'No weight decay'
        },
        {
            'name': 'gen_greedy',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --num_beams 1 --experiment_name gen_greedy',
            'desc': 'Greedy decoding (num_beams=1)'
        },
        {
            'name': 'gen_beam10',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --num_beams 10 --experiment_name gen_beam10',
            'desc': 'Large beam search (num_beams=10)'
        },
        {
            'name': 'gen_beam3_short',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --num_beams 3 --max_gen_length 256 --experiment_name gen_beam3_short',
            'desc': 'Small beam with short sequences'
        },
        {
            'name': 'lr_3e4',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --learning_rate 3e-4 --experiment_name lr_3e4',
            'desc': 'Lower learning rate (3e-4)'
        },
        {
            'name': 'lr_8e4',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --learning_rate 8e-4 --experiment_name lr_8e4',
            'desc': 'Higher learning rate (8e-4)'
        },
        {
            'name': 'lr_1e3',
            'cmd': 'python3 train_t5.py --max_n_epochs 6 --skip_test_inference --learning_rate 1e-3 --experiment_name lr_1e3',
            'desc': 'High learning rate (1e-3)'
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📋 EXPERIMENT {i}/{len(experiments)}")
        success, f1 = logger.run_experiment(exp['cmd'], exp['name'], exp['desc'])
        results.append({'name': exp['name'], 'success': success, 'f1': f1})
        
        print(f"⏱️  Pausing 10 seconds before next experiment...")
        time.sleep(10)
    
    # Final summary
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
    print("="*50)
    
    print(f"📊 RESULTS SUMMARY:")
    print(f"{'Experiment':<20} {'Status':<10} {'F1 Score'}")
    print("-" * 45)
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        f1 = result['f1'] if result['f1'] else "N/A"
        print(f"{result['name']:<20} {status:<10} {f1}")
    
    print(f"\n📁 Detailed logs saved in: experiment_logs/")
    print(f"📊 Main summary: experiment_logs/experiments_summary.log")
    
    # Find best experiment
    successful_results = [r for r in results if r['success'] and r['f1']]
    if successful_results:
        best = max(successful_results, key=lambda x: float(x['f1'].replace('%', '')))
        print(f"\n🏆 BEST EXPERIMENT: {best['name']} ({best['f1']})")

if __name__ == "__main__":
    main()