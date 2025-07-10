#!/usr/bin/env python3
"""
Simple LWE Attack Demonstration

This script demonstrates the LWE attack without user interaction,
showing how the trained models can recover the secret key.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attack_engine import LWEPartialAttack
from lwe_crypto import encrypt_message


def main():
    """Main demonstration"""
    print("=== LWE Machine Learning Attack - Simple Demo ===\n")
    
    # Check if models exist
    model_dir = "../models"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Please copy your trained models to the models/ directory")
        print("Expected structure:")
        print("  models/")
        print("    bit_model_000.h5")
        print("    bit_model_001.h5")
        print("    ...")
        print("    metadata.json")
        print("    training_data.pkl")
        return 1
    
    try:
        # Initialize partial attack (loads subset of models for speed)
        attack = LWEPartialAttack(model_dir=model_dir)
        
        # Run attack with 50 models
        results = attack.run_partial_attack(num_models=50, test_message=2)
        
        print(f"\n=== Attack Summary ===")
        print(f"Models loaded: {results['models_loaded']}")
        print(f"Secret bits predicted: {results['bits_predicted']}")
        print(f"Prediction accuracy: {results['accuracy']:.1%}")
        print(f"Average confidence: {results['avg_confidence']:.3f}")
        
        # Test multiple messages
        print(f"\n=== Testing Multiple Messages ===")
        test_messages = [0, 1, 2, 3]
        
        # Note: For this demo we'll just show the methodology
        # Full decryption requires all 512 models
        print("Note: Full message decryption requires all 512 models")
        print("This demo shows partial secret recovery with 50 models")
        
        return 0
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())