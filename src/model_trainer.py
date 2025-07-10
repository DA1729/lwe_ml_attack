"""
Model Training Module

This module handles the training of neural networks for the LWE attack.
Each model is trained to predict one bit of the secret key.
"""

import numpy as np
import os
import json
import pickle
import time
from pathlib import Path
from typing import Tuple, List, Optional

# Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

from .lwe_crypto import LWEDataGenerator, create_training_data


class LWEModelTrainer:
    """
    Trainer for LWE attack models
    """
    
    def __init__(self, n: int = 512, model_dir: str = "models"):
        """
        Initialize the model trainer
        
        Args:
            n: LWE dimension (number of models to train)
            model_dir: Directory to save trained models
        """
        self.n = n
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def create_bit_model(self) -> keras.Model:
        """
        Create a neural network to predict one secret bit
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(self.n,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_training_data(self, X: np.ndarray, secret: np.ndarray):
        """
        Save training data for later use
        
        Args:
            X: Training input vectors
            secret: Secret key
        """
        data_path = self.model_dir / "training_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({'X': X, 'secret': secret}, f)
        print(f"Training data saved to {data_path}")
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load saved training data
        
        Returns:
            Tuple of (X_train, secret)
        """
        data_path = self.model_dir / "training_data.pkl"
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['secret']
    
    def generate_training_data(self, num_samples: int = 5000, 
                             save_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using LWE implementation
        
        Args:
            num_samples: Number of LWE samples to generate
            save_data: Whether to save the generated data
            
        Returns:
            Tuple of (X_train, secret)
        """
        print(f"Generating {num_samples} LWE training samples...")
        
        generator = LWEDataGenerator()
        secret, samples = generator.generate_data(num_samples)
        X, secret = create_training_data(secret, samples)
        
        print(f"Generated training data: {X.shape}")
        print(f"Secret key length: {len(secret)}")
        
        if save_data:
            self.save_training_data(X, secret)
        
        return X, secret
    
    def train_single_model(self, X: np.ndarray, secret: np.ndarray, 
                          bit_index: int, epochs: int = 30) -> Tuple[keras.Model, float]:
        """
        Train a model for a specific bit position
        
        Args:
            X: Training input vectors
            secret: Secret key
            bit_index: Index of bit to predict
            epochs: Number of training epochs
            
        Returns:
            Tuple of (trained_model, accuracy)
        """
        # Create targets for this bit
        y_bit = np.full(len(X), secret[bit_index])
        
        # Create and train model
        model = self.create_bit_model()
        
        history = model.fit(
            X, y_bit,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            shuffle=True
        )
        
        # Evaluate accuracy
        predictions = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_bit, predictions)
        
        return model, accuracy
    
    def train_models_chunked(self, X: np.ndarray, secret: np.ndarray, 
                           chunk_size: int = 100, epochs: int = 30) -> List[float]:
        """
        Train all models in chunks for memory efficiency
        
        Args:
            X: Training input vectors
            secret: Secret key
            chunk_size: Number of models to train per chunk
            epochs: Epochs per model
            
        Returns:
            List of accuracies for each model
        """
        print(f"Training {self.n} models in chunks of {chunk_size}")
        
        accuracies = [0.0] * self.n
        total_chunks = (self.n + chunk_size - 1) // chunk_size
        
        for chunk_num in range(total_chunks):
            start_idx = chunk_num * chunk_size
            end_idx = min(start_idx + chunk_size, self.n)
            
            print(f"\n=== Training Chunk {chunk_num + 1}/{total_chunks} ===")
            print(f"Models {start_idx} to {end_idx - 1}")
            
            for i in range(start_idx, end_idx):
                print(f"Training model {i} ({i - start_idx + 1}/{end_idx - start_idx})", end='\r')
                
                try:
                    model, accuracy = self.train_single_model(X, secret, i, epochs)
                    accuracies[i] = accuracy
                    
                    # Save model
                    model_path = self.model_dir / f"bit_model_{i:03d}.h5"
                    model.save(str(model_path))
                    
                except Exception as e:
                    print(f"\nError training model {i}: {e}")
                    accuracies[i] = 0.0
            
            chunk_avg = np.mean(accuracies[start_idx:end_idx])
            print(f"\nChunk {chunk_num + 1} complete. Average accuracy: {chunk_avg:.3f}")
            
            # Save progress
            self.save_progress(accuracies, chunk_num + 1, total_chunks)
        
        print(f"\n=== Training Complete ===")
        avg_accuracy = np.mean(accuracies)
        print(f"Average accuracy across all models: {avg_accuracy:.3f}")
        
        # Save final metadata
        self.save_final_metadata(accuracies)
        
        return accuracies
    
    def save_progress(self, accuracies: List[float], current_chunk: int, total_chunks: int):
        """
        Save training progress
        
        Args:
            accuracies: List of model accuracies
            current_chunk: Current chunk number
            total_chunks: Total number of chunks
        """
        progress = {
            'trained_models': [i for i, acc in enumerate(accuracies) if acc > 0],
            'accuracies': accuracies,
            'current_chunk': current_chunk,
            'total_chunks': total_chunks,
            'n': self.n
        }
        
        with open(self.model_dir / "training_progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
    
    def save_final_metadata(self, accuracies: List[float]):
        """
        Save final metadata after training completion
        
        Args:
            accuracies: List of model accuracies
        """
        metadata = {
            'n': self.n,
            'q': 12289,
            'num_models': self.n,
            'valid_models': len([acc for acc in accuracies if acc > 0]),
            'accuracies': accuracies,
            'tensorflow_version': tf.__version__,
            'model_indices': list(range(self.n)),
            'training_completed': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Final metadata saved!")
    
    def resume_training(self, epochs: int = 30) -> List[float]:
        """
        Resume training from saved progress
        
        Args:
            epochs: Epochs per model
            
        Returns:
            Updated list of accuracies
        """
        progress_path = self.model_dir / "training_progress.json"
        if not progress_path.exists():
            raise FileNotFoundError("No training progress found to resume")
        
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        trained_models = set(progress['trained_models'])
        accuracies = progress['accuracies']
        
        # Load training data
        X, secret = self.load_training_data()
        
        # Continue training remaining models
        remaining_models = [i for i in range(self.n) if i not in trained_models]
        print(f"Resuming training for {len(remaining_models)} remaining models")
        
        for i, model_idx in enumerate(remaining_models):
            print(f"Training model {model_idx} ({i + 1}/{len(remaining_models)})", end='\r')
            
            try:
                model, accuracy = self.train_single_model(X, secret, model_idx, epochs)
                accuracies[model_idx] = accuracy
                
                # Save model
                model_path = self.model_dir / f"bit_model_{model_idx:03d}.h5"
                model.save(str(model_path))
                
                # Update progress periodically
                if (i + 1) % 10 == 0:
                    trained_models.add(model_idx)
                    self.save_progress(accuracies, -1, -1)  # -1 indicates resumed training
                    
            except Exception as e:
                print(f"\nError training model {model_idx}: {e}")
                accuracies[model_idx] = 0.0
        
        print(f"\nTraining resumed and completed!")
        
        # Save final metadata
        self.save_final_metadata(accuracies)
        
        return accuracies


def main():
    """Main training function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LWE attack models')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples to generate')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Number of models to train per chunk')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs per model')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from saved progress')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LWEModelTrainer(model_dir=args.model_dir)
    
    if args.resume:
        # Resume training
        accuracies = trainer.resume_training(epochs=args.epochs)
    else:
        # Generate training data
        X, secret = trainer.generate_training_data(num_samples=args.samples)
        
        # Train models
        accuracies = trainer.train_models_chunked(
            X, secret, 
            chunk_size=args.chunk_size, 
            epochs=args.epochs
        )
    
    print(f"Training completed with average accuracy: {np.mean(accuracies):.3f}")


if __name__ == "__main__":
    main()