# Usage Guide

This document provides detailed instructions for using the LWE ML Attack toolkit.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lwe_ml_attack.git
cd lwe_ml_attack

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Copy Your Models

Before running demos, copy your trained models to the `models/` directory:

```
models/
├── bit_model_000.h5
├── bit_model_001.h5
├── ...
├── bit_model_511.h5
├── metadata.json
└── training_data.pkl
```

### 3. Run Demonstrations

```bash
# Quick demo (20 models, ~15 seconds)
python demos/fast_demo.py

# Simple demo (50 models, ~30 seconds) 
python demos/simple_demo.py

# Full demo (512 models, ~5 minutes)
python demos/full_demo.py
```

## Detailed Usage

### Training New Models

If you need to train new models from scratch:

```bash
# Generate training data and train models
python src/model_trainer.py --samples 5000 --chunk-size 100 --epochs 30

# Resume interrupted training
python src/model_trainer.py --resume

# Train with custom parameters
python src/model_trainer.py --samples 10000 --chunk-size 50 --epochs 50 --model-dir custom_models/
```

### Using the API

```python
from src.attack_engine import LWEAttackEngine, LWEPartialAttack
from src.lwe_crypto import encrypt_message

# Quick partial attack
attack = LWEPartialAttack(model_dir="models/")
results = attack.run_partial_attack(num_models=50)

# Full attack
engine = LWEAttackEngine(model_dir="models/")
engine.load_models()
results = engine.run_full_attack()

# Custom attack
a, b = encrypt_message(message=2, secret_key=secret)
predicted_secret, confidences = engine.predict_secret_key(a)
decrypted = engine.attack_ciphertext((a, b), predicted_secret)
```

### Model Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples` | 5000 | Number of LWE samples for training |
| `--chunk-size` | 100 | Models to train per chunk |
| `--epochs` | 30 | Training epochs per model |
| `--model-dir` | models | Directory to save models |

### Demo Options

#### Fast Demo (`fast_demo.py`)
- **Purpose**: Quick validation and testing
- **Models**: 20 (configurable)
- **Runtime**: ~15 seconds
- **Shows**: Basic attack methodology

#### Simple Demo (`simple_demo.py`)
- **Purpose**: Detailed analysis with subset
- **Models**: 50 (configurable)  
- **Runtime**: ~30 seconds
- **Shows**: Secret key comparison, accuracy metrics

#### Full Demo (`full_demo.py`)
- **Purpose**: Complete attack demonstration
- **Models**: 512 (all available)
- **Runtime**: ~5 minutes
- **Shows**: Full secret recovery, message decryption

## Output Interpretation

### Secret Key Comparison
```
Bit  | Original | Predicted | Confidence | Match
------------------------------------------------
  0  |    1     |     1     |   1.000    |  ✓
  1  |    0     |     0     |   0.987    |  ✓
  2  |    1     |     0     |   0.623    |  ✗
```

- **Original**: True secret bit value
- **Predicted**: Model's prediction
- **Confidence**: Model confidence (0.5-1.0)
- **Match**: ✓ correct, ✗ incorrect

### Attack Metrics

- **Accuracy**: Percentage of correctly predicted bits
- **Confidence**: Average model confidence
- **Success Rate**: Percentage of correctly decrypted messages

### Warning Signs

🚨 **Perfect Confidence (1.000)**: May indicate overfitting
⚠️ **Low Accuracy (<70%)**: May need better training or parameters
ℹ️ **Partial Success**: Normal for subset of models

## Troubleshooting

### Common Issues

1. **"No models found"**
   - Ensure models are in `models/` directory
   - Check file naming: `bit_model_XXX.h5`

2. **"Training data not found"**
   - Copy `training_data.pkl` to models directory
   - Or regenerate with `model_trainer.py`

3. **CUDA/GPU errors**
   - Models automatically use CPU
   - Set `CUDA_VISIBLE_DEVICES='-1'` if needed

4. **Memory issues**
   - Reduce chunk size for training
   - Use partial attack demos for testing

### Performance Tips

- **Fast Testing**: Use `fast_demo.py` with 10-20 models
- **Memory Optimization**: Train in smaller chunks
- **Parallel Training**: Use multiple processes (advanced)

## File Structure

```
lwe_ml_attack/
├── src/                    # Core implementation
│   ├── lwe_crypto.py      # LWE primitives
│   ├── attack_engine.py   # Attack orchestration  
│   ├── model_trainer.py   # Training pipeline
│   └── __init__.py        # Package init
├── demos/                 # Demonstration scripts
│   ├── fast_demo.py       # Quick demo
│   ├── simple_demo.py     # Detailed demo
│   └── full_demo.py       # Complete demo
├── models/                # Trained models (user copies)
├── data/                  # Training data
├── docs/                  # Documentation
└── results/               # Output and analysis
```

## Security Notes

⚠️ **Educational Use Only**: This tool is for research and education
🔒 **Ethical Responsibility**: Users must ensure legal and ethical use
📚 **Learning Purpose**: Demonstrates cryptographic vulnerabilities
🛡️ **Real-world LWE**: Production systems use much stronger parameters

For more advanced usage and customization, see the source code documentation and examples in the `src/` directory.