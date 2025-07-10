#!/usr/bin/env python3
"""
Test single LoRA experiment for Table 2 reproduction

This script tests a single LoRA experiment to verify the setup works correctly
before running the full Table 2 reproduction.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lora_experiment():
    """Test a single LoRA experiment"""
    logger.info("🔧 Testing LoRA experiment...")
    
    # Parameters (using smaller values for quick test)
    model_name = "roberta-base"
    task_name = "sst2"
    lora_r = 8
    lora_alpha = 16
    num_train_epochs = 1  # Reduced for quick test
    learning_rate = 2e-5
    batch_size = 16  # Reduced for quick test
    max_seq_length = 128
    seed = 42
    
    # Output directory
    output_dir = Path("test_lora_results")
    output_dir.mkdir(exist_ok=True)
    
    # Training command
    cmd = [
        "uv", "run", "python", "run_minus_training.py",
        "--output_dir", str(output_dir),
        "--model_name_or_path", model_name,
        "--task_name", task_name,
        "--do_train",
        "--do_eval",
        "--apply_lora",
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--max_seq_length", str(max_seq_length),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size),
        "--num_train_epochs", str(num_train_epochs),
        "--learning_rate", str(learning_rate),
        "--save_strategy", "epoch",
        "--logging_steps", "10",
        "--seed", str(seed),
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_accuracy",
        "--greater_is_better", "True",
        "--eval_strategy", "epoch",
        "--disable_tqdm", "false",
        "--report_to", "none",
        "--max_train_samples", "1000",  # Limit for quick test
        "--max_eval_samples", "500"     # Limit for quick test
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"✅ LoRA test completed successfully in {training_time:.2f}s")
        
        # Check if expected files were created
        expected_files = [
            "train_results.json",
            "eval_results.json",
            "efficiency_results.json"
        ]
        
        results = {}
        for file_name in expected_files:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"✅ {file_name} created successfully")
                with open(file_path) as f:
                    results[file_name.replace('.json', '')] = json.load(f)
            else:
                logger.warning(f"⚠️ {file_name} not found")
        
        # Print key results
        if 'eval_results' in results:
            accuracy = results['eval_results'].get('eval_accuracy', 'N/A')
            logger.info(f"📊 Final accuracy: {accuracy}")
        
        if 'train_results' in results:
            train_runtime = results['train_results'].get('train_runtime', 'N/A')
            logger.info(f"⏱️ Training runtime: {train_runtime}s")
        
        if 'efficiency_results' in results:
            num_params = results['efficiency_results'].get('num_parameters', 'N/A')
            logger.info(f"🔢 Model parameters: {num_params:,}" if isinstance(num_params, int) else f"🔢 Model parameters: {num_params}")
        
        logger.info("🎉 LoRA test successful! Ready for full Table 2 reproduction.")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ LoRA test failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Standard output: {e.stdout}")
        return False

def main():
    logger.info("🚀 Starting LoRA test for Table 2 reproduction...")
    
    success = test_lora_experiment()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✅ TEST PASSED - Ready for full Table 2 reproduction!")
        logger.info("Next step: Run table2_roberta_sst2.py for complete experiments")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ TEST FAILED - Please fix issues before running Table 2")
        logger.error("="*60)

if __name__ == "__main__":
    main() 