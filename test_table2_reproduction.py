#!/usr/bin/env python3
"""
Test script for Table 2 reproduction

This script runs a simplified version of Table 2 experiments to verify the setup works correctly.
Uses smaller models and fewer epochs for quick testing.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import logging

# Setup logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Table2QuickTest:
    """Quick test class for Table 2 reproduction setup"""
    
    def __init__(self):
        self.output_dir = Path("test_table2_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Use roberta-base which is supported by APT framework
        self.model = "roberta-base"  # Supported by APT
        self.task = "sst2"
        self.test_methods = ["lora", "apt"]  # Test key methods only
        
    def test_environment(self):
        """Test if the environment is set up correctly"""
        logger.info("Testing environment setup...")
        
        # Check if we're in the APT directory
        if not Path("run_minus_training.py").exists():
            logger.error("Not in APT directory. Please run from APT directory.")
            return False
            
        # Test basic import
        try:
            import torch
            import transformers
            import loralib
            logger.info(f"PyTorch: {torch.__version__}")
            logger.info(f"Transformers: {transformers.__version__}")
            logger.info("Environment setup OK")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def test_efficiency_measurement(self):
        """Test the efficiency measurement functionality"""
        logger.info("Testing efficiency measurement...")
        
        # Create a simple test command that avoids CUDA issues
        test_cmd = [
            "uv", "run", "python", "-c",
            """
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Load a small model for testing
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Simple CPU-based efficiency test
device = torch.device('cpu')  # Force CPU to avoid MPS issues
model = model.to(device)

# Create dummy inputs
inputs = tokenizer("This is a test sentence", return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Time inference
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
end_time = time.time()

print('Efficiency test passed!')
print(f'Inference time: {end_time - start_time:.4f}s')
print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
"""
        ]
        
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, check=True, timeout=120)
            logger.info("Efficiency measurement test passed")
            logger.info(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Efficiency measurement test failed: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Efficiency measurement test timed out")
            return False
    
    def test_lora_training(self):
        """Test LoRA training with minimal setup"""
        logger.info("Testing LoRA training...")
        
        exp_dir = self.output_dir / "test_lora"
        exp_dir.mkdir(exist_ok=True)
        
        train_cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--model_name_or_path", self.model,
            "--task_name", self.task,
            "--output_dir", str(exp_dir),
            "--apply_lora", "true",
            "--lora_r", "4",
            "--lora_alpha", "8",
            "--per_device_train_batch_size", "8",
            "--per_device_eval_batch_size", "16",
            "--num_train_epochs", "1",  # Just 1 epoch for testing
            "--learning_rate", "2e-4",
            "--eval_strategy", "epoch",  # Fixed: was evaluation_strategy
            "--save_strategy", "epoch",
            "--logging_steps", "10",
            "--max_train_samples", "100",  # Limit training samples
            "--max_eval_samples", "50",   # Limit eval samples
            "--seed", "42",
            "--disable_tqdm", "false",
            "--use_cpu", "true",  # Force CPU to avoid MPS issues
            "--no_cuda", "true"   # Explicitly disable CUDA
        ]
        
        try:
            logger.info(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True, timeout=600)  # 10 minute timeout
            logger.info("LoRA training test passed")
            return True
        except subprocess.TimeoutExpired:
            logger.error("LoRA training test timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"LoRA training test failed: {e}")
            return False
    
    def test_apt_training(self):
        """Test APT training with minimal setup"""
        logger.info("Testing APT training...")
        
        exp_dir = self.output_dir / "test_apt"
        exp_dir.mkdir(exist_ok=True)
        
        train_cmd = [
            "uv", "run", "python", "run_minus_training.py",
            "--model_name_or_path", self.model,
            "--task_name", self.task,
            "--output_dir", str(exp_dir),
            "--apply_lora", "true",
            "--lora_r", "4",
            "--lora_alpha", "8",
            "--do_distill", "true",
            "--pruner_type", "running_fisher",
            "--pre_tuning_constraint", "0.8",  # Fixed: was constraint
            "--distillation_type", "self_interleave",
            "--pruning_scheduler", "once",
            "--param_allocation_strategy", "free_inout",
            "--per_device_train_batch_size", "8",
            "--per_device_eval_batch_size", "16",
            "--num_train_epochs", "1",  # Just 1 epoch for testing
            "--learning_rate", "2e-4",
            "--eval_strategy", "epoch",  # Fixed: was evaluation_strategy
            "--save_strategy", "epoch",
            "--logging_steps", "10",
            "--max_train_samples", "100",
            "--max_eval_samples", "50",
            "--seed", "42",
            "--disable_tqdm", "false",
            "--use_cpu", "true",  # Force CPU to avoid MPS issues
            "--no_cuda", "true"   # Explicitly disable CUDA
        ]
        
        try:
            logger.info(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True, timeout=900)  # 15 minute timeout
            logger.info("APT training test passed")
            return True
        except subprocess.TimeoutExpired:
            logger.error("APT training test timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"APT training test failed: {e}")
            return False
    
    def test_efficiency_evaluation(self):
        """Test efficiency evaluation on trained models"""
        logger.info("Testing efficiency evaluation...")
        
        # Test on LoRA model if it exists
        lora_model_path = self.output_dir / "test_lora"
        if not lora_model_path.exists():
            logger.warning("No LoRA model found for efficiency testing")
            return False
        
        # Create simple efficiency test script to avoid DeepSpeed issues
        simple_efficiency_script = """
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import json
import sys
import os

model_path = sys.argv[1]
output_dir = sys.argv[2]

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cpu')  # Force CPU
    model = model.to(device)
    
    # Simple efficiency metrics
    param_count = sum(p.numel() for p in model.parameters())
    
    # Test inference time
    test_input = "This is a test sentence for efficiency measurement."
    inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Measure time
    start_time = time.time()
    num_inferences = 10
    for _ in range(num_inferences):
        with torch.no_grad():
            outputs = model(**inputs)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_inferences
    
    results = {
        "model_path": model_path,
        "total_parameters": param_count,
        "average_inference_time_ms": avg_inference_time * 1000,
        "device": str(device)
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "efficiency_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("Efficiency evaluation completed successfully")
    print(f"Results: {results}")
    
except Exception as e:
    print(f"Error in efficiency evaluation: {e}")
    sys.exit(1)
"""
        
        # Write the script to a temp file
        temp_script = self.output_dir / "temp_efficiency_test.py"
        with open(temp_script, "w") as f:
            f.write(simple_efficiency_script)
        
        efficiency_cmd = [
            "uv", "run", "python", str(temp_script),
            str(lora_model_path),
            str(self.output_dir / "efficiency_test")
        ]
        
        try:
            subprocess.run(efficiency_cmd, check=True, timeout=180)  # 3 minute timeout
            
            # Check if results were generated
            results_file = self.output_dir / "efficiency_test" / "efficiency_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Efficiency test results: {results}")
                
                # Clean up temp script
                temp_script.unlink()
                
                logger.info("Efficiency evaluation test passed")
                return True
            else:
                logger.error("No efficiency results generated")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Efficiency evaluation test failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Efficiency evaluation test timed out")
            return False
        finally:
            # Clean up temp script if it exists
            if temp_script.exists():
                temp_script.unlink()
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        tests = [
            ("Environment", self.test_environment),
            ("Efficiency Measurement", self.test_efficiency_measurement),
            ("LoRA Training", self.test_lora_training),
            ("APT Training", self.test_apt_training),
            ("Efficiency Evaluation", self.test_efficiency_evaluation)
        ]
        
        results = {}
        passed_tests = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✓ {test_name} test PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"✗ {test_name} test FAILED")
            except Exception as e:
                logger.error(f"✗ {test_name} test FAILED with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            logger.info("🎉 All tests passed! Ready for full Table 2 reproduction.")
            return True
        else:
            logger.warning("⚠️  Some tests failed. Please fix issues before running full reproduction.")
            return False

def main():
    # Check if we're already in APT directory or need to change
    if not Path("run_minus_training.py").exists():
        if Path("APT/run_minus_training.py").exists():
            logger.info("Changed to APT directory")
            os.chdir("APT")
        else:
            logger.error("Could not find APT directory with run_minus_training.py")
            logger.error("Please run this script from the repository root or APT directory")
            sys.exit(1)
    
    logger.info("Starting Table 2 reproduction tests")
    
    test_runner = Table2QuickTest()
    success = test_runner.run_all_tests()
    
    if success:
        print("\n" + "="*60)
        print("READY FOR FULL REPRODUCTION")
        print("="*60)
        print("All tests passed. You can now run the full Table 2 reproduction:")
        print("  python table2_reproduction.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SETUP ISSUES DETECTED")
        print("="*60)
        print("Please fix the failed tests before proceeding.")
        print("="*60)

if __name__ == "__main__":
    main() 