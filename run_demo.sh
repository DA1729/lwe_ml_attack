#!/bin/bash

# LWE ML Attack Demo Runner
# This script sets up and runs the LWE attack demonstrations

echo "=== LWE ML Attack Demo Runner ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if models directory exists and has models
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "Models directory is empty. Copying models..."
    if [ -f "copy_models.py" ]; then
        python3 copy_models.py
    else
        echo "Error: copy_models.py not found"
        echo "Please manually copy your models to the models/ directory"
        exit 1
    fi
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt --quiet
fi

echo ""
echo "Available demos:"
echo "1. Fast Demo (20 models, ~15 seconds)"
echo "2. Simple Demo (50 models, ~30 seconds)"  
echo "3. Full Demo (512 models, ~5 minutes)"
echo "4. Exit"
echo ""

while true; do
    read -p "Select demo to run (1-4): " choice
    case $choice in
        1)
            echo ""
            echo "Running Fast Demo..."
            python3 demos/fast_demo.py
            break
            ;;
        2)
            echo ""
            echo "Running Simple Demo..."
            python3 demos/simple_demo.py
            break
            ;;
        3)
            echo ""
            echo "Running Full Demo (this may take several minutes)..."
            python3 demos/full_demo.py
            break
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select 1-4."
            ;;
    esac
done

echo ""
echo "Demo completed!"