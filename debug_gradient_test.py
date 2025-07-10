#!/usr/bin/env python3
"""
Debug Gradient Test - Investigate gradient flow issue
"""

import sys
import os
sys.path.insert(0, 'APT')

import torch
from transformers import RobertaTokenizer, RobertaConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gradient_debug():
    """Debug the gradient flow issue"""
    logger.info("Testing gradient flow...")
    
    try:
        # Import APT components
        from models.modeling_roberta import CoFiRobertaForSequenceClassification
        from utils.minus_utils import linear_to_lora
        from loralib import mark_only_lora_as_trainable
        
        # Create model config
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = 2
        config.apply_lora = True
        config.lora_r = 8
        config.lora_alpha = 16
        config.do_distill = False
        
        # Create model
        model = CoFiRobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            config=config
        )
        
        # Convert MORE layers to LoRA (not just one)
        logger.info("Converting multiple layers to LoRA...")
        first_layer = model.roberta.encoder.layer[0]
        
        # Convert query, key, and value projections
        for layer_name in ['query', 'key', 'value']:
            layer = getattr(first_layer.attention.self, layer_name)
            lora_layer = linear_to_lora(layer, r=config.lora_r, lora_alpha=config.lora_alpha)
            setattr(first_layer.attention.self, layer_name, lora_layer)
            logger.info(f"Converted {layer_name} to LoRA")
        
        # Also convert the classifier layers
        classifier_dense = model.classifier.dense
        lora_classifier = linear_to_lora(classifier_dense, r=config.lora_r, lora_alpha=config.lora_alpha)
        model.classifier.dense = lora_classifier
        logger.info("Converted classifier.dense to LoRA")
        
        # Mark only LoRA parameters as trainable
        mark_only_lora_as_trainable(model)
        
        # Check trainable parameters
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(trainable_params)}")
        
        # Create inputs
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        inputs = tokenizer("This is a test sentence.", return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = torch.tensor([1])
        
        # Forward pass with more detailed monitoring
        logger.info("Forward pass...")
        outputs = model(**inputs)
        loss = outputs.loss
        
        logger.info(f"Loss: {loss.item():.4f}")
        logger.info(f"Loss requires_grad: {loss.requires_grad}")
        logger.info(f"Loss grad_fn: {loss.grad_fn}")
        
        # Check if loss has computational graph
        if loss.grad_fn is None:
            logger.error("Loss has no grad_fn! This is the problem.")
            
            # Let's see what the forward pass is producing
            logger.info("Inspecting model outputs...")
            logger.info(f"Output keys: {outputs.keys()}")
            logger.info(f"Logits shape: {outputs.logits.shape}")
            logger.info(f"Logits requires_grad: {outputs.logits.requires_grad}")
            logger.info(f"Logits grad_fn: {outputs.logits.grad_fn}")
            
            return False
        
        # Try backward pass
        logger.info("Backward pass...")
        loss.backward()
        
        # Check gradients
        grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
        logger.info(f"Parameters with gradients: {len(grad_params)}")
        
        return len(grad_params) > 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting Gradient Debug Test...")
    logger.info("=" * 50)
    
    success = test_gradient_debug()
    
    logger.info("=" * 50)
    logger.info(f"Gradient Debug Test: {'PASSED' if success else 'FAILED'}")
