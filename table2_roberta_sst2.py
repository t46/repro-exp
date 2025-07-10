#!/usr/bin/env python3
"""
Table 2 Reproduction: RoBERTa-base + SST2 experiments

This script reproduces Table 2 experiments for RoBERTa-base model on SST2 task.
All methods are tested at 60% sparsity (constraint=0.4, meaning 40% of parameters remain).

Methods tested:
- FT (Fine-tuning)
- LoRA
- LoRA+Prune  
- Prune+Distill
- LoRA+Prune+Distill
- APT

Metrics measured:
- Task accuracy
- Training Time (97% TTA)
- Training Memory
- Inference Time  
- Inference Memory
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Table2RoBERTaSST2Reproducer:
    """Reproduces Table 2 experiments for RoBERTa-base + SST2"""
    
    def __init__(self):
        self.model_name = "roberta-base"
        self.task_name = "sst2"
        self.constraint = 0.4  # 60% sparsity = 40% parameters remain
        self.lora_r = 8
        self.lora_alpha = 16
        self.num_train_epochs = 5
        self.learning_rate = 2e-5
        self.batch_size = 32
        self.max_seq_length = 128
        self.seed = 42
        
        # Output directory
        self.output_base = Path("table2_roberta_sst2_results")
        self.output_base.mkdir(exist_ok=True)
        
        # Common training arguments
        self.common_args = [
            "--model_name_or_path", self.model_name,
            "--task_name", self.task_name,
            "--do_train",
            "--do_eval", 
            "--max_seq_length", str(self.max_seq_length),
            "--per_device_train_batch_size", str(self.batch_size),
            "--per_device_eval_batch_size", str(self.batch_size),
            "--num_train_epochs", str(self.num_train_epochs),
            "--learning_rate", str(self.learning_rate),
            "--evaluation_strategy", "epoch",
            "--save_strategy", "epoch",
            "--logging_steps", "100",
            "--seed", str(self.seed),
            "--load_best_model_at_end",
            "--metric_for_best_model", "eval_accuracy",
            "--greater_is_better", "True",
            "--disable_tqdm", "false",
            "--report_to", "none"
        ]
        
        self.results = {}
        
    def run_fine_tuning(self) -> Dict[str, Any]:
        """Run standard fine-tuning baseline"""
        logger.info("🔥 Running Fine-Tuning (FT) baseline...")
        
        output_dir = self.output_base / "ft"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            *self.common_args
        ]
        
        return self._run_experiment("FT", cmd, output_dir)
    
    def run_lora(self) -> Dict[str, Any]:
        """Run LoRA baseline"""
        logger.info("🔧 Running LoRA baseline...")
        
        output_dir = self.output_base / "lora"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            "--apply_lora",
            "--lora_r", str(self.lora_r),
            "--lora_alpha", str(self.lora_alpha),
            *self.common_args
        ]
        
        return self._run_experiment("LoRA", cmd, output_dir)
    
    def run_lora_prune(self) -> Dict[str, Any]:
        """Run LoRA+Prune baseline"""
        logger.info("✂️ Running LoRA+Prune baseline...")
        
        output_dir = self.output_base / "lora_prune"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            "--apply_lora",
            "--lora_r", str(self.lora_r),
            "--lora_alpha", str(self.lora_alpha),
            "--mac_constraint", str(self.constraint),
            "--pruner_type", "running_fisher",
            "--pruning_scheduler", "once",
            "--param_allocation_strategy", "free_inout",
            *self.common_args
        ]
        
        return self._run_experiment("LoRA+Prune", cmd, output_dir)
    
    def run_prune_distill(self) -> Dict[str, Any]:
        """Run Prune+Distill baseline"""
        logger.info("🎓 Running Prune+Distill baseline...")
        
        output_dir = self.output_base / "prune_distill"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            "--do_distill",
            "--mac_constraint", str(self.constraint),
            "--pruner_type", "running_fisher",
            "--pruning_scheduler", "once",
            "--distillation_type", "self_interleave",
            *self.common_args
        ]
        
        return self._run_experiment("Prune+Distill", cmd, output_dir)
    
    def run_lora_prune_distill(self) -> Dict[str, Any]:
        """Run LoRA+Prune+Distill baseline"""
        logger.info("🔧✂️🎓 Running LoRA+Prune+Distill baseline...")
        
        output_dir = self.output_base / "lora_prune_distill"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            "--apply_lora",
            "--lora_r", str(self.lora_r),
            "--lora_alpha", str(self.lora_alpha),
            "--do_distill",
            "--mac_constraint", str(self.constraint),
            "--pruner_type", "running_fisher",
            "--pruning_scheduler", "once",
            "--distillation_type", "self_interleave",
            "--param_allocation_strategy", "free_inout",
            *self.common_args
        ]
        
        return self._run_experiment("LoRA+Prune+Distill", cmd, output_dir)
    
    def run_apt(self) -> Dict[str, Any]:
        """Run APT (our method)"""
        logger.info("🚀 Running APT (our method)...")
        
        output_dir = self.output_base / "apt"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--output_dir", str(output_dir),
            "--apply_lora",
            "--lora_r", str(self.lora_r),
            "--lora_alpha", str(self.lora_alpha),
            "--do_distill",
            "--mac_constraint", str(self.constraint),
            "--pruner_type", "running_fisher",
            "--pruning_scheduler", "once",
            "--distillation_type", "self_interleave",
            "--param_allocation_strategy", "free_inout",
            # APT-specific parameters
            "--constraint", str(1 - self.constraint),  # 60% sparsity
            *self.common_args
        ]
        
        return self._run_experiment("APT", cmd, output_dir)
    
    def _run_experiment(self, method_name: str, cmd: List[str], output_dir: Path) -> Dict[str, Any]:
        """Run a single experiment and collect results"""
        logger.info(f"Running {method_name} experiment...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run the training
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info(f"✅ {method_name} completed successfully in {training_time:.2f}s")
            
            # Parse results
            results = self._parse_results(method_name, output_dir, training_time)
            
            # Save individual results
            with open(output_dir / "summary.json", "w") as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {method_name} failed with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            return {
                "method": method_name,
                "status": "failed",
                "error": str(e),
                "training_time": None,
                "accuracy": None,
                "efficiency_metrics": {}
            }
    
    def _parse_results(self, method_name: str, output_dir: Path, training_time: float) -> Dict[str, Any]:
        """Parse training and efficiency results"""
        results = {
            "method": method_name,
            "status": "success",
            "training_time": training_time
        }
        
        # Parse training results
        train_results_file = output_dir / "train_results.json"
        if train_results_file.exists():
            with open(train_results_file) as f:
                train_results = json.load(f)
                results["train_runtime"] = train_results.get("train_runtime", training_time)
                results["train_samples_per_second"] = train_results.get("train_samples_per_second")
        
        # Parse evaluation results  
        eval_results_file = output_dir / "eval_results.json"
        if eval_results_file.exists():
            with open(eval_results_file) as f:
                eval_results = json.load(f)
                results["accuracy"] = eval_results.get("eval_accuracy")
        
        # Parse efficiency results
        efficiency_file = output_dir / "efficiency_results.json"
        if efficiency_file.exists():
            with open(efficiency_file) as f:
                efficiency_results = json.load(f)
                results["efficiency_metrics"] = efficiency_results
        
        # Parse run report
        run_report_file = output_dir / "run_report.json"
        if run_report_file.exists():
            with open(run_report_file) as f:
                run_report = json.load(f)
                results["peak_memory"] = run_report.get("peak_mem")
                results["train_runtime_per_epoch"] = run_report.get("train_runtime_per_epoch")
        
        return results
    
    def run_all_experiments(self):
        """Run all Table 2 experiments for RoBERTa + SST2"""
        logger.info("🚀 Starting Table 2 reproduction for RoBERTa-base + SST2...")
        
        experiments = [
            ("FT", self.run_fine_tuning),
            ("LoRA", self.run_lora),
            ("LoRA+Prune", self.run_lora_prune),
            ("Prune+Distill", self.run_prune_distill),
            ("LoRA+Prune+Distill", self.run_lora_prune_distill),
            ("APT", self.run_apt)
        ]
        
        for method_name, experiment_func in experiments:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {method_name}")
            logger.info(f"{'='*60}")
            
            try:
                results = experiment_func()
                self.results[method_name] = results
                
                # Log summary
                accuracy = results.get("accuracy", "N/A")
                training_time = results.get("training_time", "N/A")
                logger.info(f"✅ {method_name}: Accuracy={accuracy}, Time={training_time}s")
                
            except Exception as e:
                logger.error(f"❌ {method_name} failed: {e}")
                self.results[method_name] = {
                    "method": method_name,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a summary report comparing all methods"""
        logger.info("\n" + "="*80)
        logger.info("TABLE 2 REPRODUCTION SUMMARY - RoBERTa-base + SST2")
        logger.info("="*80)
        
        # Create summary table
        summary_data = []
        for method_name, results in self.results.items():
            if results.get("status") == "success":
                efficiency = results.get("efficiency_metrics", {})
                row = {
                    "Method": method_name,
                    "Accuracy": f"{results.get('accuracy', 0):.3f}" if results.get('accuracy') else "N/A",
                    "Training Time (s)": f"{results.get('training_time', 0):.1f}" if results.get('training_time') else "N/A",
                    "Training Memory (MB)": f"{results.get('peak_memory', 0):.1f}" if results.get('peak_memory') else "N/A",
                    "Inference Time (ms)": f"{efficiency.get('bz32_t_mean', 0)*1000:.1f}" if efficiency.get('bz32_t_mean') else "N/A",
                    "Inference Memory (MB)": f"{efficiency.get('bz32_em_mean', 0):.1f}" if efficiency.get('bz32_em_mean') else "N/A",
                    "Parameters": f"{efficiency.get('num_parameters', 0):,}" if efficiency.get('num_parameters') else "N/A"
                }
            else:
                row = {
                    "Method": method_name,
                    "Accuracy": "FAILED",
                    "Training Time (s)": "FAILED", 
                    "Training Memory (MB)": "FAILED",
                    "Inference Time (ms)": "FAILED",
                    "Inference Memory (MB)": "FAILED",
                    "Parameters": "FAILED"
                }
            summary_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        
        # Print to console
        print("\n")
        print(df.to_string(index=False))
        
        # Save to files
        summary_file = self.output_base / "table2_summary.csv"
        df.to_csv(summary_file, index=False)
        
        results_file = self.output_base / "table2_full_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n📊 Summary saved to: {summary_file}")
        logger.info(f"📄 Full results saved to: {results_file}")

def main():
    reproducer = Table2RoBERTaSST2Reproducer()
    reproducer.run_all_experiments()

if __name__ == "__main__":
    main() 