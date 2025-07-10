#!/usr/bin/env python3
"""
Simple test script without DeepSpeed dependencies
"""

import os
import sys
import torch
from pathlib import Path

# Add APT to Python path
current_dir = Path(__file__).parent.absolute()
apt_dir = current_dir / "APT"
sys.path.insert(0, str(apt_dir))

def test_basic_imports():
    """Test basic imports without DeepSpeed"""
    print("Testing basic imports...")
    
    try:
        import loralib
        print("✓ loralib imported successfully")
    except ImportError as e:
        print(f"✗ loralib import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        from utils.minus_utils import efficiency_testing
        print("✓ efficiency_testing imported successfully")
    except ImportError as e:
        print(f"✗ efficiency_testing import failed: {e}")
        return False
    
    return True

def test_simple_model():
    """Test loading and using a simple model"""
    print("\nTesting simple model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load a small model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        print(f"✓ {model_name} loaded successfully")
        
        # Test forward pass
        text = "This is a test sentence."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        
        print(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_lora_functionality():
    """Test LoRA functionality"""
    print("\nTesting LoRA functionality...")
    
    try:
        import loralib as lora
        
        # Create a simple LoRA layer
        layer = lora.Linear(768, 256, r=8, lora_alpha=16)
        
        # Test forward pass
        x = torch.randn(1, 768)
        output = layer(x)
        
        print(f"✓ LoRA layer test successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ LoRA test failed: {e}")
        return False

def main():
    print("=== Simple APT Environment Test ===")
    
    # Check if we're already in APT directory or need to change
    if Path("run_minus_training.py").exists():
        print("Already in APT directory")
    elif Path("APT/run_minus_training.py").exists():
        os.chdir("APT")
        print("Changed to APT directory")
    else:
        print("Cannot find APT directory with required files")
        sys.exit(1)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Simple Model", test_simple_model),
        ("LoRA Functionality", test_lora_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, status in results.items():
        print(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic environment is working!")
    else:
        print("⚠️ Some issues detected.")
    
    return passed == total

if __name__ == "__main__":
    main() 