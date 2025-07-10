#!/usr/bin/env python3
"""
APT Integration Test - Following the actual APT workflow
"""

import sys
import os
sys.path.insert(0, 'APT')

import torch
from transformers import RobertaTokenizer
import logging
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_apt_integration():
    """Test APT's full integration workflow"""
    logger.info("Testing APT integration workflow...")
    
    try:
        # Import APT components
        from models.modeling_roberta import CoFiRobertaForSequenceClassification
        from models.model_args import ModelArguments
        from args import MinusTrainingArguments
        from trainer.param_control import ParamController
        from trainer.model_arch import ModelArch
        from loralib import mark_only_lora_as_trainable
        
        logger.info("APT imports successful")
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        logger.info("Tokenizer loaded")
        
        # Load config and set APT-specific parameters first
        from transformers import RobertaConfig
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = 2
        config.finetuning_task = "sst2"
        config.apply_lora = True
        config.lora_r = 8
        config.lora_alpha = 16
        config.do_distill = False
        
        # Create model with configured LoRA settings
        model = CoFiRobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            config=config
        )
        
        logger.info("Model loaded with LoRA config")
        
        # Create training arguments
        training_args = MinusTrainingArguments(
            output_dir="./apt_test",
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            learning_rate=2e-4,
        )
        
        # Create model arguments
        model_args = ModelArguments(model_name_or_path="roberta-base")
        
        # Initialize param controller and model architecture
        model_arch = ModelArch(model)
        param_controller = ParamController(
            model=model,
            teacher_config=None,
            student_config=None,
            warmup_config={0: {'query': 'teacher', 'key': 'teacher', 'value': 'teacher'}, 1: {'query': 'teacher'}},  # Basic warmup config
            param_config={},
            model_arch=model_arch,
            teacher_tuning_config=None,
            lora_with_bias=False
        )
        
        logger.info("ParamController initialized")
        
        # Test the layer conversion process (this is key!)
        logger.info("Converting layers to LoRA...")
        param_controller.convert_to_pre_pruning_lora_teacher()
        param_controller.model_as_teacher()
        
        # Now mark LoRA parameters as trainable
        mark_only_lora_as_trainable(model)
        
        # Check trainable parameters
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters after LoRA conversion: {len(trainable_params)}")
        
        if len(trainable_params) == 0:
            logger.error("No trainable parameters after LoRA conversion!")
            return False
            
        # Show some examples
        for i, name in enumerate(trainable_params[:5]):
            logger.info(f"  {i+1}. {name}")
        if len(trainable_params) > 5:
            logger.info(f"  ... and {len(trainable_params) - 5} more")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        inputs = tokenizer("This is a test sentence.", return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = torch.tensor([1])
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        logger.info(f"Forward pass successful. Loss: {loss.item():.4f}")
        
        # Test backward pass
        logger.info("Testing backward pass...")
        loss.backward()
        
        # Check gradients
        grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
        logger.info(f"Parameters with gradients: {len(grad_params)}")
        
        if len(grad_params) == 0:
            logger.error("No gradients computed!")
            return False
        else:
            logger.info("Backward pass successful!")
            
        # Show gradient examples
        for i, name in enumerate(grad_params[:3]):
            param = dict(model.named_parameters())[name]
            logger.info(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in APT integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting APT Integration Test...")
    logger.info("=" * 60)
    
    success = test_apt_integration()
    
    logger.info("=" * 60)
    logger.info(f"APT Integration Test: {'PASSED' if success else 'FAILED'}")
    
    if success:
        logger.info("✅ APT layer conversion and gradient computation working correctly!")
    else:
        logger.info("❌ APT integration has issues that need to be resolved.") 