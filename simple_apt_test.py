#!/usr/bin/env python3
"""
Simplified APT Test - Focus on core LoRA layer conversion
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

def test_simple_lora_conversion():
    """Test direct LoRA layer conversion without complex param controller"""
    logger.info("Testing simplified LoRA conversion...")
    
    try:
        # Import APT components
        from models.modeling_roberta import CoFiRobertaForSequenceClassification
        from utils.minus_utils import linear_to_lora
        from loralib import mark_only_lora_as_trainable
        
        logger.info("APT imports successful")
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        logger.info("Tokenizer loaded")
        
        # Create model config with APT parameters
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
        logger.info("Model loaded")
        
        # Manually convert some key layers to LoRA (following APT pattern)
        logger.info("Converting query layer in first attention head to LoRA...")
        
        # Get the first layer's query projection
        first_layer = model.roberta.encoder.layer[0]
        query_layer = first_layer.attention.self.query
        
        logger.info(f"Original query layer type: {type(query_layer)}")
        
        # Convert to LoRA using APT's utility function
        lora_query = linear_to_lora(
            query_layer, 
            r=config.lora_r, 
            lora_alpha=config.lora_alpha
        )
        
        # Replace the layer
        first_layer.attention.self.query = lora_query
        
        logger.info(f"Converted query layer type: {type(lora_query)}")
        logger.info(f"LoRA layer has r={lora_query.r}")
        
        # Now mark only LoRA parameters as trainable
        mark_only_lora_as_trainable(model)
        
        # Check trainable parameters after LoRA conversion
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters after LoRA conversion: {len(trainable_params)}")
        
        if len(trainable_params) == 0:
            logger.error("No trainable parameters after LoRA conversion!")
            return False
            
        # Show some trainable parameter names
        for i, name in enumerate(trainable_params[:5]):
            logger.info(f"  {i+1}. {name}")
        
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
            
        return True
        
    except Exception as e:
        logger.error(f"Error in simplified LoRA test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting Simplified APT LoRA Test...")
    logger.info("=" * 70)
    
    success = test_simple_lora_conversion()
    
    logger.info("=" * 70)
    logger.info(f"Simplified APT LoRA Test: {'PASSED' if success else 'FAILED'}")
    
    if success:
        logger.info("✅ Core LoRA conversion working!")
    else:
        logger.info("❌ Core LoRA conversion has issues.")
