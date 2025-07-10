#!/usr/bin/env python3
"""
Full LWE Attack Demonstration

Complete demonstration of the LWE attack using all 512 trained models
for full secret key recovery and message decryption.
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attack_engine import LWEAttackEngine


def main():
    """Main demonstration"""
    print("LWE Machine Learning Attack - Full Demo")
    print("=" * 50)
    print("⚠️  Warning: This demo loads all 512 models and may take several minutes")
    
    model_dir = "../models" 
    
    if not os.path.exists(model_dir):
        print(f"\nError: Model directory '{model_dir}' not found!")
        print("Please copy your trained models to the models/ directory")
        return 1
    
    try:
        # Initialize full attack engine
        attack = LWEAttackEngine(model_dir=model_dir)
        
        print(f"\nLoading all trained models...")
        start_time = time.time()
        
        # Load all models
        loaded_count = attack.load_models()
        load_time = time.time() - start_time
        
        if loaded_count == 0:
            print("Error: No models could be loaded!")
            return 1
            
        print(f"✓ Loaded {loaded_count} models in {load_time:.1f}s")
        
        if loaded_count < 512:
            print(f"⚠️  Warning: Only {loaded_count}/512 models loaded")
            print("   Attack may not achieve full secret recovery")
        
        # Run complete attack demonstration
        print(f"\nRunning full attack demonstration...")
        results = attack.run_full_attack(test_messages=[0, 1, 2, 3])
        
        # Display comprehensive results
        print(f"\n" + "="*60)
        print(f"FINAL ATTACK RESULTS")
        print(f"="*60)
        
        print(f"Models loaded: {loaded_count}/512 ({loaded_count/512:.1%})")
        print(f"Success rate: {results['success_rate']:.1%}")
        print(f"Messages decrypted: {results['successful_decryptions']}/{results['total_tests']}")
        
        # Analyze individual test results
        print(f"\n=== Detailed Results ===")
        for i, test in enumerate(results['test_results']):
            status = "✓ SUCCESS" if test['success'] else "✗ FAILED"
            print(f"Message {test['message']}: {test['decrypted']} {status}")
            if 'metrics' in test:
                metrics = test['metrics']
                print(f"  Secret accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"  Avg confidence: {metrics.get('avg_confidence', 0):.3f}")
        
        # Security implications
        print(f"\n=== Security Analysis ===")
        if results['success_rate'] >= 0.75:
            print("🚨 CRITICAL: LWE encryption is completely broken!")
            print("   - Secret key successfully recovered")
            print("   - All messages can be decrypted")
            print("   - Cryptographic security is compromised")
        elif results['success_rate'] >= 0.25:
            print("⚠️  WARNING: Partial attack success")
            print("   - Some secret bits recovered")
            print("   - Cryptographic weakness demonstrated")
        else:
            print("ℹ️  Attack had limited success")
            print("   - May need parameter adjustment or more training")
        
        print(f"\n=== Implications ===")
        print("• This demonstrates potential ML threats to lattice-based crypto")
        print("• Real-world LWE uses much stronger parameters")
        print("• Proper parameter selection is critical for security")
        print("• Consider post-quantum cryptographic standards (NIST)")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running full demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())