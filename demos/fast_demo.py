#!/usr/bin/env python3
"""
Fast LWE Attack Demo

Quick demonstration of the LWE attack using a small subset of models
for rapid testing and validation.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attack_engine import LWEPartialAttack


def main():
    """Main demo function"""
    print("LWE Attack - Fast Demo")
    print("=" * 50)
    
    model_dir = "../models"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Please copy your trained models to the models/ directory")
        return 1
    
    try:
        # Initialize attack with 20 models for speed
        attack = LWEPartialAttack(model_dir=model_dir)
        
        print("Running fast attack demonstration...")
        print("Loading 20 models for quick testing...\n")
        
        # Run attack
        results = attack.run_partial_attack(num_models=20, test_message=1)
        
        # Show results summary
        print(f"\n=== Fast Demo Results ===")
        print(f"✓ Models loaded: {results['models_loaded']}")
        print(f"✓ Secret bits predicted: {results['bits_predicted']}")  
        print(f"✓ Accuracy: {results['accuracy']:.1%}")
        print(f"✓ Average confidence: {results['avg_confidence']:.3f}")
        
        if results['accuracy'] >= 0.9:
            print("\n🎯 Attack appears successful on loaded models!")
        else:
            print("\n⚠️  Lower accuracy - may need more training or different parameters")
        
        print(f"\n=== Scaling Analysis ===")
        print(f"Current performance: {results['bits_predicted']}/512 bits ({results['bits_predicted']/512:.1%})")
        print(f"With all 512 models: Complete secret recovery expected")
        print(f"Full attack time estimate: ~{20 * 512 / results['models_loaded']:.0f}x longer")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())