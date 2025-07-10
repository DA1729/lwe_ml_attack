"""
LWE Attack Engine

This module implements the main attack orchestration for breaking LWE using
machine learning models. It coordinates model loading, prediction, and
secret key recovery.
"""

import numpy as np
import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

from .lwe_crypto import LWECrypto, encrypt_message, decrypt_message


class LWEAttackEngine:
    """
    Main engine for orchestrating LWE attacks using machine learning models
    """
    
    def __init__(self, model_dir: str = "models", n: int = 512, q: int = 12289):
        """
        Initialize the attack engine
        
        Args:
            model_dir: Directory containing trained models
            n: LWE dimension
            q: LWE modulus
        """
        self.n = n
        self.q = q
        self.model_dir = Path(model_dir)
        self.models = [None] * n
        self.metadata = None
        self.crypto = LWECrypto(n, q)
        
    def load_models(self, max_models: Optional[int] = None) -> int:
        """
        Load trained models from disk
        
        Args:
            max_models: Maximum number of models to load (None for all)
            
        Returns:
            Number of models successfully loaded
        """
        print("=== Loading Pre-trained Models ===")
        
        # Load metadata if available
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Found metadata for {self.metadata['num_models']} models")
        
        # Determine how many models to load
        num_to_load = min(max_models or self.n, self.n)
        
        # Load models
        loaded_count = 0
        failed_models = []
        
        for i in range(num_to_load):
            model_path = self.model_dir / f"bit_model_{i:03d}.h5"
            if model_path.exists():
                try:
                    self.models[i] = keras.models.load_model(str(model_path))
                    loaded_count += 1
                    if loaded_count % 10 == 0:
                        print(f"Loaded {loaded_count} models...", end='\r')
                except Exception as e:
                    print(f"Warning: Could not load model {i}: {e}")
                    failed_models.append(i)
            else:
                failed_models.append(i)
        
        print(f"Successfully loaded {loaded_count}/{num_to_load} models")
        if failed_models and len(failed_models) < 10:
            print(f"Failed to load models: {failed_models}")
        elif failed_models:
            print(f"Failed to load {len(failed_models)} models")
            
        return loaded_count
    
    def load_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load original training data if available
        
        Returns:
            Tuple of (X_train, secret_key) or (None, None) if not found
        """
        try:
            data_path = self.model_dir / "training_data.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data['X'], data['secret']
        except FileNotFoundError:
            print("Warning: Original training data not found")
            return None, None
    
    def predict_secret_key(self, sample_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the secret key using loaded models
        
        Args:
            sample_a: LWE vector a for prediction
            
        Returns:
            Tuple of (predicted_secret, confidences)
        """
        if not any(model is not None for model in self.models):
            raise ValueError("No models loaded! Call load_models() first.")
        
        predicted_secret = []
        confidences = []
        
        # Reshape for neural network input
        input_data = sample_a.reshape(1, -1)
        
        for i, model in enumerate(self.models):
            if model is not None:
                try:
                    prob = model.predict(input_data, verbose=0)[0][0]
                    bit_prediction = 1 if prob > 0.5 else 0
                    confidence = max(prob, 1 - prob)
                except Exception as e:
                    print(f"Warning: Error predicting with model {i}: {e}")
                    bit_prediction = 0
                    confidence = 0.5
            else:
                # If model not available, random guess
                bit_prediction = np.random.randint(0, 2)
                confidence = 0.5
                
            predicted_secret.append(bit_prediction)
            confidences.append(confidence)
            
        return np.array(predicted_secret), np.array(confidences)
    
    def predict_secret_bits_subset(self, sample_a: np.ndarray) -> Tuple[Dict[int, int], Dict[int, float]]:
        """
        Predict secret bits using only loaded models
        
        Args:
            sample_a: LWE vector a for prediction
            
        Returns:
            Tuple of (predicted_bits_dict, confidences_dict)
        """
        predicted_bits = {}
        confidences = {}
        
        input_data = sample_a.reshape(1, -1)
        
        for i, model in enumerate(self.models):
            if model is not None:
                try:
                    prob = model.predict(input_data, verbose=0)[0][0]
                    bit_prediction = 1 if prob > 0.5 else 0
                    confidence = max(prob, 1 - prob)
                    
                    predicted_bits[i] = bit_prediction
                    confidences[i] = confidence
                    
                except Exception as e:
                    print(f"Warning: Error predicting with model {i}: {e}")
        
        return predicted_bits, confidences
    
    def attack_ciphertext(self, ciphertext: Tuple[np.ndarray, int], 
                         predicted_secret: np.ndarray) -> int:
        """
        Decrypt a ciphertext using predicted secret key
        
        Args:
            ciphertext: LWE ciphertext (a, b)
            predicted_secret: Predicted secret key
            
        Returns:
            Decrypted message
        """
        return self.crypto.decrypt(ciphertext, predicted_secret)
    
    def evaluate_attack(self, original_secret: np.ndarray, predicted_secret: np.ndarray,
                       confidences: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate attack performance
        
        Args:
            original_secret: Original secret key
            predicted_secret: Predicted secret key
            confidences: Model confidences
            
        Returns:
            Dictionary of evaluation metrics
        """
        if original_secret is None:
            return {"error": "No original secret available for evaluation"}
        
        correct_bits = np.sum(original_secret == predicted_secret)
        total_bits = len(original_secret)
        accuracy = correct_bits / total_bits
        
        # Analyze confidence distribution
        high_conf_bits = np.sum(confidences >= 0.9)
        low_conf_bits = np.sum(confidences < 0.7)
        avg_confidence = np.mean(confidences)
        
        return {
            'correct_bits': int(correct_bits),
            'total_bits': int(total_bits),
            'accuracy': float(accuracy),
            'avg_confidence': float(avg_confidence),
            'high_confidence_bits': int(high_conf_bits),
            'low_confidence_bits': int(low_conf_bits),
            'perfect_predictions': int(np.sum(confidences >= 0.999))
        }
    
    def display_secret_comparison(self, original_secret: np.ndarray, 
                                predicted_secret: np.ndarray, confidences: np.ndarray,
                                num_display: int = 30) -> Tuple[int, int]:
        """
        Display detailed comparison between original and predicted secret bits
        
        Args:
            original_secret: Original secret key
            predicted_secret: Predicted secret key
            confidences: Model confidences
            num_display: Number of bits to display
            
        Returns:
            Tuple of (correct_count, total_count)
        """
        print(f"\n=== Secret Key Comparison (First {num_display} bits) ===")
        print("Bit  | Original | Predicted | Confidence | Match")
        print("-" * 48)
        
        correct_count = 0
        total_count = min(num_display, len(original_secret))
        
        for i in range(total_count):
            original_bit = original_secret[i]
            predicted_bit = predicted_secret[i]
            confidence = confidences[i]
            match = "✓" if original_bit == predicted_bit else "✗"
            
            if original_bit == predicted_bit:
                correct_count += 1
                
            print(f"{i:3d}  |    {original_bit}     |     {predicted_bit}     |   {confidence:.3f}    |  {match}")
        
        accuracy = correct_count / total_count
        print(f"\nAccuracy: {correct_count}/{total_count} ({accuracy:.1%})")
        
        # Show side-by-side comparison
        print(f"\n=== Side-by-Side Secret Key Comparison ===")
        print(f"Original  (bits 0-{num_display-1}): {' '.join(str(original_secret[i]) for i in range(num_display))}")
        print(f"Predicted (bits 0-{num_display-1}): {' '.join(str(predicted_secret[i]) for i in range(num_display))}")
        
        # Show match status
        match_display = ['✓' if original_secret[i] == predicted_secret[i] else '✗' for i in range(num_display)]
        print(f"Match     (bits 0-{num_display-1}): {' '.join(match_display)}")
        
        return correct_count, total_count
    
    def run_full_attack(self, test_messages: List[int] = [0, 1, 2, 3]) -> Dict[str, Any]:
        """
        Run complete attack demonstration
        
        Args:
            test_messages: Messages to test decryption on
            
        Returns:
            Attack results and statistics
        """
        print("=== LWE Machine Learning Attack ===\n")
        
        # Load training data
        X_train, original_secret = self.load_training_data()
        if original_secret is None:
            raise ValueError("Cannot run attack without original secret for comparison")
        
        print(f"Original secret (first 20 bits): {original_secret[:20]}")
        
        results = {
            'successful_decryptions': 0,
            'total_tests': len(test_messages),
            'test_results': [],
            'overall_metrics': None
        }
        
        for message in test_messages:
            print(f"\n--- Testing Message: {message} ---")
            
            # Encrypt message
            a, b = encrypt_message(message, original_secret)
            print(f"Encrypted: a=(vector), b={b}")
            
            # Predict secret key
            start_time = time.time()
            predicted_secret, confidences = self.predict_secret_key(a)
            prediction_time = time.time() - start_time
            
            # Decrypt message
            decrypted_message = self.attack_ciphertext((a, b), predicted_secret)
            success = decrypted_message == message
            
            if success:
                results['successful_decryptions'] += 1
            
            print(f"Predicted secret key in {prediction_time:.3f}s")
            print(f"Average model confidence: {np.mean(confidences):.3f}")
            print(f"Decrypted message: {decrypted_message}")
            print(f"Decryption {'SUCCESS' if success else 'FAILED'}")
            
            # Evaluate attack
            metrics = self.evaluate_attack(original_secret, predicted_secret, confidences)
            
            # Show comparison
            correct_bits, total_bits = self.display_secret_comparison(
                original_secret, predicted_secret, confidences, num_display=20
            )
            
            results['test_results'].append({
                'message': message,
                'decrypted': decrypted_message,
                'success': success,
                'prediction_time': prediction_time,
                'metrics': metrics
            })
        
        # Overall results
        success_rate = results['successful_decryptions'] / results['total_tests']
        print(f"\n=== Final Attack Results ===")
        print(f"Successfully decrypted {results['successful_decryptions']}/{results['total_tests']} messages")
        print(f"Overall success rate: {success_rate:.1%}")
        
        results['success_rate'] = success_rate
        return results


class LWEPartialAttack:
    """Attack engine for partial secret recovery with subset of models"""
    
    def __init__(self, model_dir: str = "models", n: int = 512, q: int = 12289):
        self.attack_engine = LWEAttackEngine(model_dir, n, q)
        
    def run_partial_attack(self, num_models: int = 50, 
                          test_message: int = 2) -> Dict[str, Any]:
        """
        Run attack with subset of models
        
        Args:
            num_models: Number of models to load
            test_message: Message to test
            
        Returns:
            Attack results
        """
        print(f"=== LWE Partial Attack ({num_models} models) ===\n")
        
        # Load subset of models
        loaded_count = self.attack_engine.load_models(max_models=num_models)
        if loaded_count == 0:
            raise ValueError("No models could be loaded!")
        
        # Load training data
        X_train, original_secret = self.attack_engine.load_training_data()
        if original_secret is None:
            raise ValueError("Cannot run attack without original secret")
        
        print(f"Original secret (first {min(30, num_models)} bits): {original_secret[:min(30, num_models)]}")
        
        # Encrypt test message
        a, b = encrypt_message(test_message, original_secret)
        print(f"Testing with message: {test_message}")
        print(f"Encrypted: a=(vector), b={b}")
        
        # Predict available secret bits
        predicted_bits, confidences = self.attack_engine.predict_secret_bits_subset(a)
        
        print(f"\nPredicted {len(predicted_bits)} secret bits")
        
        # Show comparison
        print(f"\n=== Detailed Secret Key Comparison ===")
        print("Bit | Original | Predicted | Confidence | Match")
        print("-" * 45)
        
        correct_predictions = 0
        for i in sorted(predicted_bits.keys()):
            original_bit = original_secret[i]
            predicted_bit = predicted_bits[i]
            confidence = confidences[i]
            is_correct = original_bit == predicted_bit
            
            if is_correct:
                correct_predictions += 1
                
            status = "✓" if is_correct else "✗"
            print(f"{i:2d}  |    {original_bit}     |     {predicted_bit}     |   {confidence:.3f}    |  {status}")
        
        accuracy = correct_predictions / len(predicted_bits)
        print(f"\nPartial secret recovery accuracy: {correct_predictions}/{len(predicted_bits)} ({accuracy:.1%})")
        
        return {
            'models_loaded': loaded_count,
            'bits_predicted': len(predicted_bits),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_confidence': np.mean(list(confidences.values())),
            'predicted_bits': predicted_bits,
            'confidences': confidences
        }