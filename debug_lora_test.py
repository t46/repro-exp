#!/usr/bin/env python3
"""
Simple LoRA debugging test to identify core issues
"""

import sys
import os
sys.path.insert(0, 'APT')

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import TrainingArguments
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_lora():
    """Test basic LoRA integration with RoBERTa"""
    logger.info("Testing basic LoRA functionality...")
    
    try:
        # Load model and tokenizer
        logger.info("Loading RoBERTa model...")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", 
            num_labels=2,
            finetuning_task="sst2"
        )
        
        # Apply LoRA manually
        logger.info("Applying LoRA layers...")
        from loralib import mark_only_lora_as_trainable, Linear as LoRALinear
        
        # Set LoRA config
        model.config.apply_lora = True
        lora_r = 8
        lora_alpha = 16
        
        # Check original trainable parameters
        orig_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Original trainable parameters: {len(orig_trainable)}")
        
        # Mark LoRA parameters as trainable (this actually marks non-LoRA as non-trainable)
        mark_only_lora_as_trainable(model)
        
        logger.info("Model with LoRA layers created successfully")
        
        # Check which parameters require gradients
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(trainable_params)}")
        for i, name in enumerate(trainable_params[:5]):  # Show first 5
            logger.info(f"  {i+1}. {name}")
        if len(trainable_params) > 5:
            logger.info(f"  ... and {len(trainable_params) - 5} more")
            
        # Create a simple input
        logger.info("Testing forward pass...")
        inputs = tokenizer("This is a test sentence.", return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = torch.tensor([1])  # Positive label
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        logger.info(f"Forward pass successful. Loss: {loss.item():.4f}")
        
        # Test backward pass
        logger.info("Testing backward pass...")
        loss.backward()
        logger.info("Backward pass successful")
        
        # Check if gradients are computed
        grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
        logger.info(f"Parameters with gradients: {len(grad_params)}")
        
        if len(grad_params) == 0:
            logger.error("No gradients computed! This is the issue.")
            return False
        else:
            logger.info("Gradients computed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error in LoRA test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_apt_framework():
    """Test APT framework components"""
    logger.info("Testing APT framework components...")
    
    try:
        # Try importing specific components to avoid circular imports
        logger.info("Testing basic APT imports...")
        
        # Test basic model args first
        from models.model_args import ModelArguments
        model_args = ModelArguments(model_name_or_path="roberta-base")
        logger.info("ModelArguments import successful")
        
        # Test training args
        from args import MinusTrainingArguments
        training_args = MinusTrainingArguments(
            output_dir="./debug_test",
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            learning_rate=2e-4,
        )
        logger.info("MinusTrainingArguments import successful")
        
        # Test LoRA layers directly  
        from loralib.layers import Linear as LoRALinear, PruningLinear
        logger.info("LoRA layers import successful")
        
        logger.info("APT framework basic imports successful")
        return True
        
    except Exception as e:
        logger.error(f"Error in APT framework test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting APT debugging tests...")
    
    # Test 1: Basic LoRA functionality
    logger.info("=" * 50)
    logger.info("TEST 1: Basic LoRA functionality")
    logger.info("=" * 50)
    lora_success = test_basic_lora()
    
    # Test 2: APT framework components
    logger.info("=" * 50)
    logger.info("TEST 2: APT framework components")
    logger.info("=" * 50)
    apt_success = test_apt_framework()
    
    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"LoRA test: {'PASSED' if lora_success else 'FAILED'}")
    logger.info(f"APT framework test: {'PASSED' if apt_success else 'FAILED'}")
    
    if lora_success and apt_success:
        logger.info("All tests passed! The issue may be in the integration.")
    else:
        logger.info("Some tests failed. Debugging needed.") 