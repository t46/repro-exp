#!/usr/bin/env python3
"""
Focused Table 2 Reproduction Script for APT Paper

This script reproduces key experiments from Table 2 focusing on the main methods:
- LoRA baseline
- APT method
- Fine-tuning baseline

Targets RoBERTa-base on SST2 first, then expands to other tasks/models.
"""

import os
import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedTable2Reproducer:
    """Focused reproducer for Table 2 key experiments"""
    
    def __init__(self, output_dir: str = "focused_table2_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Start with core model and task
        self.model = "roberta-base"
        self.task = "sst2"
        
        # Core methods to compare
        self.methods = {
            "lora": "LoRA",
            "apt": "APT"
        }
        
        self.results = {}
        
        # Training configurations
        self.base_config = {
            "do_train": "true",  # Enable training
            "do_eval": "true",   # Enable evaluation
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": 5,
            "learning_rate": 2e-4,
            "eval_strategy": "epoch",  # Fixed: was evaluation_strategy
            "save_strategy": "epoch", 
            "logging_steps": 50,
            "seed": 42,
            "max_train_samples": 2000,  # Manageable size for testing
            "max_eval_samples": 500,
            "disable_tqdm": "false",
            "max_grad_norm": 0.1  # Add gradient clipping to prevent numerical overflow
        }
    
    def run_lora_experiment(self) -> Dict[str, Any]:
        """Run LoRA baseline experiment"""
        logger.info("Starting LoRA experiment")
        
        exp_dir = self.output_dir / "lora_sst2"
        exp_dir.mkdir(exist_ok=True)
        
        train_cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--model_name_or_path", self.model,
            "--task_name", self.task,
            "--output_dir", str(exp_dir),
            "--apply_lora", "true",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--teacher_param_tuning_config", "q:0-11,v:0-11"  # Required for LoRA layer creation
        ]
        
        # Add base configuration
        for key, value in self.base_config.items():
            train_cmd.extend([f"--{key}", str(value)])
        
        try:
            logger.info(f"Running command: {' '.join(train_cmd)}")
            start_time = time.time()
            
            # Run with output capture for better debugging - change to APT directory
            result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800,  # 30 minute timeout
                cwd='APT'  # Run from APT directory
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode != 0:
                logger.error(f"LoRA training failed with return code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                return {"error": "Training failed", "stderr": result.stderr}
            
            logger.info(f"LoRA training completed in {training_time:.2f} seconds")
            
            # Extract results
            results = self._extract_training_results(exp_dir)
            results["training_wall_time"] = training_time
            
            # Run efficiency test
            efficiency_results = self._run_efficiency_test(exp_dir)
            results.update(efficiency_results)
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error("LoRA training timed out")
            return {"error": "Training timed out"}
        except Exception as e:
            logger.error(f"LoRA training failed: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run_apt_experiment(self) -> Dict[str, Any]:
        """Run APT experiment"""
        logger.info("Starting APT experiment")
        
        exp_dir = self.output_dir / "apt_sst2"
        exp_dir.mkdir(exist_ok=True)
        
        train_cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--model_name_or_path", self.model,
            "--task_name", self.task,
            "--output_dir", str(exp_dir),
            "--apply_lora", "true",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--teacher_param_tuning_config", "q:0-11,v:0-11",  # Required for LoRA layer creation
            "--do_distill", "true",
            "--pruner_type", "running_fisher",
            "--pre_tuning_constraint", "0.8",  # 20% pruning for testing
            "--distillation_type", "self_interleave",
            "--pruning_scheduler", "once",
            "--param_allocation_strategy", "free_inout"
        ]
        
        # Add base configuration
        for key, value in self.base_config.items():
            train_cmd.extend([f"--{key}", str(value)])
        
        try:
            logger.info(f"Running command: {' '.join(train_cmd)}")
            start_time = time.time()
            
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=2400,  # 40 minute timeout for APT
                cwd='APT'  # Run from APT directory
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode != 0:
                logger.error(f"APT training failed with return code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                return {"error": "Training failed", "stderr": result.stderr}
            
            logger.info(f"APT training completed in {training_time:.2f} seconds")
            
            # Extract results
            results = self._extract_training_results(exp_dir)
            results["training_wall_time"] = training_time
            
            # Run efficiency test
            efficiency_results = self._run_efficiency_test(exp_dir)
            results.update(efficiency_results)
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error("APT training timed out")
            return {"error": "Training timed out"}
        except Exception as e:
            logger.error(f"APT training failed: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _extract_training_results(self, exp_dir: Path) -> Dict[str, Any]:
        """Extract training results from experiment directory"""
        results = {}
        
        # Check for trainer state
        trainer_state_file = exp_dir / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_data = json.load(f)
                    
                log_history = trainer_data.get("log_history", [])
                if log_history:
                    # Get final eval results
                    eval_entries = [entry for entry in log_history if "eval_accuracy" in entry]
                    if eval_entries:
                        final_eval = eval_entries[-1]
                        results["eval_accuracy"] = final_eval.get("eval_accuracy", 0)
                        results["eval_loss"] = final_eval.get("eval_loss", 0)
                        results["epoch"] = final_eval.get("epoch", 0)
                
                # Get training time
                results["total_train_runtime"] = trainer_data.get("train_runtime", 0)
                        
            except Exception as e:
                logger.warning(f"Could not parse trainer_state.json: {e}")
        
        # Check for eval_results
        eval_results_file = exp_dir / "eval_results.json"
        if eval_results_file.exists():
            try:
                with open(eval_results_file, 'r') as f:
                    eval_data = json.load(f)
                    results.update(eval_data)
            except Exception as e:
                logger.warning(f"Could not parse eval_results.json: {e}")
        
        return results
    
    def _run_efficiency_test(self, exp_dir: Path) -> Dict[str, Any]:
        """Run efficiency test on trained model"""
        logger.info(f"Running efficiency test for {exp_dir}")
        
        efficiency_cmd = [
            "uv", "run", "python", "efficiency_test.py",
            "--model_name_or_path", str(exp_dir),
            "--task_name", self.task,
            "--output_dir", str(exp_dir),
            "--per_device_eval_batch_size", "32"
        ]
        
        try:
            result = subprocess.run(
                efficiency_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd='APT'  # Run from APT directory
            )
            
            if result.returncode != 0:
                logger.warning(f"Efficiency test failed: {result.stderr}")
                return {"efficiency_error": result.stderr}
            
            # Load efficiency results
            efficiency_file = exp_dir / "efficiency_results.json"
            if efficiency_file.exists():
                with open(efficiency_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Efficiency results file not found")
                return {}
                
        except subprocess.TimeoutExpired:
            logger.warning("Efficiency test timed out")
            return {"efficiency_error": "Timeout"}
        except Exception as e:
            logger.warning(f"Efficiency test failed: {e}")
            return {"efficiency_error": str(e)}
    
    def run_all_experiments(self):
        """Run all focused experiments"""
        logger.info("Starting focused Table 2 reproduction")
        
        # Run LoRA experiment
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: LoRA Baseline")
        logger.info("=" * 60)
        lora_results = self.run_lora_experiment()
        self.results["lora"] = lora_results
        self._save_intermediate_results()
        
        # Run APT experiment
        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: APT Method")
        logger.info("=" * 60)
        apt_results = self.run_apt_experiment()
        self.results["apt"] = apt_results
        self._save_intermediate_results()
        
        # Generate comparison report
        self._generate_comparison_report()
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        results_file = self.output_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Intermediate results saved to {results_file}")
    
    def _generate_comparison_report(self):
        """Generate comparison report"""
        logger.info("Generating comparison report")
        
        report = []
        report.append("=" * 80)
        report.append("FOCUSED TABLE 2 REPRODUCTION RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Performance comparison
        report.append("PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        for method, results in self.results.items():
            if "error" in results:
                report.append(f"{method.upper()}: FAILED - {results['error']}")
            else:
                accuracy = results.get("eval_accuracy", 0) * 100
                loss = results.get("eval_loss", 0)
                train_time = results.get("training_wall_time", 0)
                report.append(f"{method.upper()}:")
                report.append(f"  Accuracy: {accuracy:.2f}%")
                report.append(f"  Loss: {loss:.4f}")
                report.append(f"  Training Time: {train_time:.2f}s")
        
        report.append("")
        
        # Efficiency comparison
        report.append("EFFICIENCY COMPARISON:")
        report.append("-" * 40)
        
        lora_results = self.results.get("lora", {})
        apt_results = self.results.get("apt", {})
        
        if not lora_results.get("error") and not apt_results.get("error"):
            # Compare training times
            lora_time = lora_results.get("training_wall_time", 0)
            apt_time = apt_results.get("training_wall_time", 0)
            
            if lora_time > 0:
                speedup = lora_time / apt_time if apt_time > 0 else 0
                report.append(f"APT Training Speedup: {speedup:.2f}x")
            
            # Compare accuracies
            lora_acc = lora_results.get("eval_accuracy", 0) * 100
            apt_acc = apt_results.get("eval_accuracy", 0) * 100
            acc_diff = apt_acc - lora_acc
            report.append(f"APT vs LoRA Accuracy: {acc_diff:+.2f}%")
            
            # Memory efficiency (if available)
            lora_mem = lora_results.get("bz32_em_mean", 0)
            apt_mem = apt_results.get("bz32_em_mean", 0)
            if lora_mem > 0 and apt_mem > 0:
                mem_ratio = apt_mem / lora_mem
                report.append(f"APT Memory Ratio: {mem_ratio:.2f}x LoRA")
        
        report.append("")
        report.append("=" * 80)
        
        # Save and display report
        report_text = "\n".join(report)
        report_file = self.output_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info(f"Comparison report saved to {report_file}")
        
        # Save final results
        final_results_file = self.output_dir / "final_results.json"
        with open(final_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Final results saved to {final_results_file}")


def main():
    reproducer = FocusedTable2Reproducer()
    reproducer.run_all_experiments()


if __name__ == "__main__":
    main() 