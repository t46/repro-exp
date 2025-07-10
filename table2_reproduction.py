#!/usr/bin/env python3
"""
Table 2 Reproduction Script for APT Paper

This script reproduces Table 2 from the APT paper:
"APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"

Table 2 compares efficiency metrics for RoBERTa-base and T5-base models at 60% sparsity.

Methods compared:
- FT (Fine-tuning)
- LoRA
- LoRA+Prune
- Prune+Distill (RoBERTa only)
- LoRA+Prune+Distill (RoBERTa only)  
- APT

Metrics measured:
- Task accuracy (MNLI, SST2, SQuAD v2 for RoBERTa; MNLI, SST2, CNN/DM for T5)
- Training Time (97% TTA - Time to Accuracy)
- Training Memory 
- Inference Time
- Inference Memory
"""

import os
import json
import subprocess
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Table2Reproducer:
    """Main class for reproducing Table 2 experiments"""
    
    def __init__(self, output_dir: str = "table2_results", models: List[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = models or ["roberta-base", "t5-base"]
        self.target_sparsity = 0.6  # 60% sparsity as mentioned in Table 2
        
        # Task configurations
        self.roberta_tasks = ["mnli", "sst2", "squad_v2"]
        self.t5_tasks = ["mnli", "sst2", "cnn_dm"]
        
        # Methods to evaluate
        self.methods = {
            "ft": "Fine-tuning",
            "lora": "LoRA", 
            "lora_prune": "LoRA+Prune",
            "prune_distill": "Prune+Distill",  # RoBERTa only
            "lora_prune_distill": "LoRA+Prune+Distill",  # RoBERTa only
            "apt": "APT"
        }
        
        self.results = {}
        
    def run_experiment(self, model: str, method: str, task: str) -> Dict[str, Any]:
        """Run a single experiment for given model, method, and task"""
        logger.info(f"Running {method} experiment for {model} on {task}")
        
        # Create experiment directory
        exp_dir = self.output_dir / f"{model}_{method}_{task}"
        exp_dir.mkdir(exist_ok=True)
        
        if method == "ft":
            return self._run_fine_tuning(model, task, exp_dir)
        elif method == "lora":
            return self._run_lora(model, task, exp_dir)
        elif method == "lora_prune":
            return self._run_lora_prune(model, task, exp_dir)
        elif method == "prune_distill" and model == "roberta-base":
            return self._run_prune_distill(model, task, exp_dir)
        elif method == "lora_prune_distill" and model == "roberta-base":
            return self._run_lora_prune_distill(model, task, exp_dir)
        elif method == "apt":
            return self._run_apt(model, task, exp_dir)
        else:
            logger.warning(f"Method {method} not supported for model {model}")
            return {}
    
    def _run_fine_tuning(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run standard fine-tuning baseline"""
        # Fine-tuning command
        train_cmd = self._get_base_training_command(model, task, exp_dir)
        train_cmd.extend([
            "--apply_lora", "false"
        ])
        
        # Run training and measure efficiency
        return self._execute_training_and_measure(train_cmd, exp_dir, "ft")
    
    def _run_lora(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run LoRA baseline"""
        train_cmd = self._get_base_training_command(model, task, exp_dir)
        train_cmd.extend([
            "--apply_lora", "true",
            "--lora_r", "8",
            "--lora_alpha", "16"
        ])
        
        return self._execute_training_and_measure(train_cmd, exp_dir, "lora")
    
    def _run_lora_prune(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run LoRA+Prune baseline using Mask Tuning"""
        # First run LoRA training
        lora_results = self._run_lora(model, task, exp_dir / "lora_stage")
        
        # Then apply Mask Tuning pruning
        prune_cmd = [
            "uv", "run", "python", "post_training_prune.py",
            "--model_name_or_path", str(exp_dir / "lora_stage"),
            "--task_name", task,
            "--constraint", str(self.target_sparsity),
            "--output_dir", str(exp_dir / "pruned"),
            "--pruner_type", "mask_tuning"
        ]
        
        # Run pruning
        self._execute_command(prune_cmd)
        
        # Measure efficiency on pruned model
        efficiency_cmd = [
            "uv", "run", "python", "efficiency_test.py",
            "--model_name_or_path", str(exp_dir / "pruned"),
            "--task_name", task,
            "--output_dir", str(exp_dir)
        ]
        
        efficiency_results = self._execute_command(efficiency_cmd, capture_output=True)
        return self._parse_efficiency_results(efficiency_results, exp_dir)
    
    def _run_prune_distill(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run Prune+Distill baseline using CoFi"""
        train_cmd = self._get_base_training_command(model, task, exp_dir)
        train_cmd.extend([
            "--apply_lora", "false",
            "--do_distill", "true",
            "--pruner_type", "l0_regularization",
            "--constraint", str(self.target_sparsity),
            "--distillation_type", "cofi"
        ])
        
        return self._execute_training_and_measure(train_cmd, exp_dir, "prune_distill")
    
    def _run_lora_prune_distill(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run LoRA+Prune+Distill baseline"""
        train_cmd = self._get_base_training_command(model, task, exp_dir)
        train_cmd.extend([
            "--apply_lora", "true",
            "--lora_r", "8", 
            "--lora_alpha", "16",
            "--do_distill", "true",
            "--pruner_type", "l0_regularization", 
            "--constraint", str(self.target_sparsity),
            "--distillation_type", "cofi"
        ])
        
        return self._execute_training_and_measure(train_cmd, exp_dir, "lora_prune_distill")
    
    def _run_apt(self, model: str, task: str, exp_dir: Path) -> Dict[str, Any]:
        """Run APT method"""
        train_cmd = self._get_base_training_command(model, task, exp_dir)
        train_cmd.extend([
            "--apply_lora", "true",
            "--lora_r", "8",
            "--lora_alpha", "16", 
            "--do_distill", "true",
            "--pruner_type", "running_fisher",
            "--constraint", str(self.target_sparsity),
            "--distillation_type", "self_interleave",
            "--pruning_scheduler", "once",
            "--param_allocation_strategy", "free_inout"
        ])
        
        return self._execute_training_and_measure(train_cmd, exp_dir, "apt")
    
    def _get_base_training_command(self, model: str, task: str, exp_dir: Path) -> List[str]:
        """Get base training command"""
        # Select appropriate training script based on model type
        if "t5" in model:
            script = "run_minus_seq2seq_training.py"
        elif task == "squad_v2":
            script = "run_minus_squad_training.py"
        else:
            script = "run_minus_training.py"
        
        return [
            "uv", "run", "python", script,
            "--model_name_or_path", model,
            "--task_name", task,
            "--output_dir", str(exp_dir),
            "--per_device_train_batch_size", "32",
            "--per_device_eval_batch_size", "128", 
            "--num_train_epochs", "30",
            "--learning_rate", "2e-4",
            "--evaluation_strategy", "epoch",
            "--save_strategy", "epoch",
            "--logging_steps", "100",
            "--seed", "42",
            "--dataloader_num_workers", "4",
            "--disable_tqdm", "false"
        ]
    
    def _execute_training_and_measure(self, train_cmd: List[str], exp_dir: Path, method: str) -> Dict[str, Any]:
        """Execute training and measure efficiency"""
        # Run training
        logger.info(f"Executing training command: {' '.join(train_cmd)}")
        self._execute_command(train_cmd)
        
        # Measure efficiency
        efficiency_cmd = [
            "uv", "run", "python", "efficiency_test.py",
            "--model_name_or_path", str(exp_dir / "best_model"),
            "--task_name", train_cmd[train_cmd.index("--task_name") + 1],
            "--output_dir", str(exp_dir)
        ]
        
        efficiency_results = self._execute_command(efficiency_cmd, capture_output=True)
        return self._parse_efficiency_results(efficiency_results, exp_dir)
    
    def _execute_command(self, cmd: List[str], capture_output: bool = False):
        """Execute shell command"""
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return result.stdout
            else:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e}")
            if capture_output and e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if capture_output and e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            raise
    
    def _parse_efficiency_results(self, output: str, exp_dir: Path) -> Dict[str, Any]:
        """Parse efficiency measurement results"""
        # Load efficiency results from JSON file
        efficiency_file = exp_dir / "efficiency_results.json"
        if efficiency_file.exists():
            with open(efficiency_file, 'r') as f:
                efficiency_data = json.load(f)
        else:
            efficiency_data = {}
        
        # Load training metrics from trainer logs
        trainer_state_file = exp_dir / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r') as f:
                trainer_data = json.load(f)
                # Extract training time and accuracy metrics
                training_metrics = self._extract_training_metrics(trainer_data)
                efficiency_data.update(training_metrics)
        
        return efficiency_data
    
    def _extract_training_metrics(self, trainer_data: Dict) -> Dict[str, Any]:
        """Extract training time and accuracy metrics"""
        log_history = trainer_data.get("log_history", [])
        
        # Find 97% accuracy point for TTA calculation
        target_accuracy = 0.97
        training_time_97 = None
        
        for entry in log_history:
            if "eval_accuracy" in entry and entry["eval_accuracy"] >= target_accuracy:
                training_time_97 = entry.get("train_runtime", 0)
                break
        
        return {
            "training_time_97_accuracy": training_time_97,
            "total_training_time": trainer_data.get("train_runtime", 0),
            "final_accuracy": log_history[-1].get("eval_accuracy", 0) if log_history else 0
        }
    
    def run_all_experiments(self):
        """Run all experiments for Table 2 reproduction"""
        logger.info("Starting Table 2 reproduction experiments")
        
        for model in self.models:
            tasks = self.roberta_tasks if model == "roberta-base" else self.t5_tasks
            
            for task in tasks:
                for method in self.methods.keys():
                    # Skip certain method-model combinations as per paper
                    if method in ["prune_distill", "lora_prune_distill"] and model != "roberta-base":
                        continue
                    
                    try:
                        results = self.run_experiment(model, method, task)
                        
                        # Store results
                        key = f"{model}_{method}_{task}"
                        self.results[key] = {
                            "model": model,
                            "method": method,
                            "task": task,
                            **results
                        }
                        
                        # Save intermediate results
                        self._save_results()
                        
                    except Exception as e:
                        logger.error(f"Experiment {model}_{method}_{task} failed: {e}")
                        continue
        
        logger.info("All experiments completed")
        self._generate_table2()
    
    def _save_results(self):
        """Save results to JSON file"""
        results_file = self.output_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _generate_table2(self):
        """Generate Table 2 from collected results"""
        logger.info("Generating Table 2")
        
        # Create table structure
        table_data = []
        
        for model in self.models:
            model_section = {"Model": f"{model.title().replace('-', ' ')}", "Method": "", 
                           "MNLI": "", "SST2": "", "SQuAD v2/CNN-DM": "",
                           "Train Time(↓)": "", "Train Mem(↓)": "", 
                           "Inf Time(↓)": "", "Inf Mem(↓)": ""}
            table_data.append(model_section)
            
            # Get baseline metrics for normalization (Fine-tuning)
            ft_key = f"{model}_ft_sst2"  # Use SST2 for efficiency baseline
            ft_results = self.results.get(ft_key, {})
            
            for method in self.methods.keys():
                # Skip methods not applicable for this model
                if method in ["prune_distill", "lora_prune_distill"] and model != "roberta-base":
                    continue
                
                method_row = {"Method": self.methods[method]}
                
                # Add task accuracies
                tasks = self.roberta_tasks if model == "roberta-base" else self.t5_tasks
                for task in tasks:
                    key = f"{model}_{method}_{task}"
                    result = self.results.get(key, {})
                    accuracy = result.get("final_accuracy", 0)
                    
                    if task == "mnli":
                        method_row["MNLI"] = f"{accuracy:.1f}" if accuracy > 0 else "-"
                    elif task == "sst2":
                        method_row["SST2"] = f"{accuracy:.1f}" if accuracy > 0 else "-"
                    elif task in ["squad_v2", "cnn_dm"]:
                        if task == "squad_v2":
                            method_row["SQuAD v2/CNN-DM"] = f"{accuracy:.1f}" if accuracy > 0 else "-"
                        else:  # CNN/DM returns ROUGE scores
                            rouge_scores = result.get("rouge_scores", {})
                            rouge_str = "/".join([f"{rouge_scores.get(f'rouge{i}', 0):.1f}" 
                                                for i in [1, 2, "L"]])
                            method_row["SQuAD v2/CNN-DM"] = rouge_str if rouge_scores else "-"
                
                # Add efficiency metrics (normalized to FT baseline)
                efficiency_key = f"{model}_{method}_sst2"  # Use SST2 for efficiency
                efficiency_result = self.results.get(efficiency_key, {})
                
                if ft_results:
                    # Calculate relative metrics
                    train_time_rel = self._calculate_relative_metric(
                        efficiency_result.get("training_time_97_accuracy", 0),
                        ft_results.get("training_time_97_accuracy", 1)
                    )
                    train_mem_rel = self._calculate_relative_metric(
                        efficiency_result.get("bz128_em_mean", 0),
                        ft_results.get("bz128_em_mean", 1)
                    )
                    inf_time_rel = self._calculate_relative_metric(
                        efficiency_result.get("bz128_t_mean", 0),
                        ft_results.get("bz128_t_mean", 1)
                    )
                    inf_mem_rel = self._calculate_relative_metric(
                        efficiency_result.get("bz128_sm_mean", 0),
                        ft_results.get("bz128_sm_mean", 1)
                    )
                    
                    method_row["Train Time(↓)"] = f"{train_time_rel:.1f}%" if train_time_rel > 0 else "-"
                    method_row["Train Mem(↓)"] = f"{train_mem_rel:.1f}%" if train_mem_rel > 0 else "-"
                    method_row["Inf Time(↓)"] = f"{inf_time_rel:.1f}%" if inf_time_rel > 0 else "-"
                    method_row["Inf Mem(↓)"] = f"{inf_mem_rel:.1f}%" if inf_mem_rel > 0 else "-"
                else:
                    method_row.update({
                        "Train Time(↓)": "-", "Train Mem(↓)": "-",
                        "Inf Time(↓)": "-", "Inf Mem(↓)": "-"
                    })
                
                table_data.append(method_row)
        
        # Save as CSV and display
        df = pd.DataFrame(table_data)
        csv_file = self.output_dir / "table2_reproduction.csv"
        df.to_csv(csv_file, index=False)
        
        # Display table
        print("\n" + "="*100)
        print("TABLE 2 REPRODUCTION RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        logger.info(f"Table 2 saved to {csv_file}")
    
    def _calculate_relative_metric(self, value: float, baseline: float) -> float:
        """Calculate relative metric as percentage of baseline"""
        if baseline == 0:
            return 0
        return (value / baseline) * 100


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 from APT paper")
    parser.add_argument("--output-dir", default="table2_results", 
                       help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=["roberta-base", "t5-base"],
                       help="Models to evaluate")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with reduced epochs")
    
    args = parser.parse_args()
    
    # Change to APT directory
    os.chdir("APT")
    
    reproducer = Table2Reproducer(args.output_dir, args.models)
    
    if args.quick_test:
        logger.info("Running quick test mode")
        # Modify training epochs for quick testing
        reproducer.quick_test = True
    
    reproducer.run_all_experiments()


if __name__ == "__main__":
    main() 