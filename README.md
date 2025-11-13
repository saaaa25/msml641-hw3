# NLP Homework 3: Comparative Analysis of RNN Architectures for Sentiment Classification

## Overview

This notebook implements and evaluates multiple sequence models (RNN, LSTM, BiLSTM) with various activation functions, optimizers, and hyperparameters on the IMDB sentiment classification dataset using GPU acceleration. The project compares model performance across different architectural configurations and generates comprehensive experimental results.

## Setup Instructions

### Python Version & Environment Requirements

- **Python Version**: 3.12.12 (or compatible 3.8+)
- **Environment**: Recommended to run on Google Colab with GPU support

### Running on Google Colab

1. **Upload the notebook to Colab**:
   - Go to https://colab.research.google.com/
   - Click `File` → `Upload notebook`
   - Select `Satwika_Konda_121331717_HW3.ipynb`

2. **Enable GPU acceleration (Optional but recommended)**:
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` under "Hardware accelerator"
   - Click `Save`

3. **Install dependencies** (automatic in notebook):
   - The first cell runs: `pip install torch tensorflow scikit-learn pandas matplotlib seaborn`
   - All required packages are installed automatically on first execution

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.8.0+cu126 | Deep learning framework with CUDA support |
| TensorFlow | 2.19.0 | Dataset utilities (IMDB dataset) |
| scikit-learn | 1.6.1 | Evaluation metrics (accuracy, F1) |
| pandas | 2.2.2 | Data manipulation and results storage |
| NumPy | 2.0.2 | Numerical computations |
| matplotlib | 3.10.0 | Visualization |
| seaborn | 0.13.2 | Advanced plotting |

## Expected Runtime and Output

### Runtime Estimates

| Component | GPU (Tesla T4) | CPU | Notes |
|-----------|----------------|-----|-------|
| Setup & Data Loading | 1-2 min | 2-3 min | One-time, includes package installation |
| Model Training (all configurations) | 15-20 min | 3-4 hrs | Depends on sequence length & batch size |
Recommended: Use GPU for faster execution 

### Output Files

The notebook generates the following files:

1. **`experiment_results.csv`** (Main Results File)
   - Columns: Model, Activation, Optimizer, Seq_Length, Grad_Clipping, Accuracy, F1, Epoch_Time, train_losses
   - Rows: 25 different model configurations
   - Size: ~50 KB

2. **`full_results.pkl`** (Pickled Results)
   - Complete results dictionary with full training histories
   - Includes loss curves for each epoch
   - Size: ~100-200 KB

3. **Console Output**:
   - Training progress for each epoch
   - Final metrics (accuracy, F1) for each configuration
   - Hardware specification summary

## Hardware Requirements

### Minimum Requirements
- **GPU**: 2 GB VRAM (for training)
- **RAM**: 4 GB
- **Storage**: 1 GB free space

### Recommended Requirements
- **GPU**: Tesla T4 or better (15 GB VRAM) - as used in experiments
- **RAM**: 13+ GB
- **Storage**: 2 GB for results and checkpoints

### Colab Specifications (Current Setup)
- **GPU**: Tesla T4 (15.83 GB)
- **CPU**: x86_64
- **RAM**: 13.61 GB
- **CUDA**: 12.6
- **OS**: Linux 6.6.105+

## Model Configurations

The notebook tests **25 different configurations** across:

**Architectures**: RNN, LSTM, BiLSTM

**Activations**: relu, sigmoid, tanh

**Optimizers**: ADAM, SGD, RMSPROP

**Sequence Lengths**: 25, 50, 100

**Gradient Clipping**: Yes/No

### Fixed Hyperparameters
- Vocabulary Size: 10,000
- Embedding Dimension: 100
- Hidden Size: 64
- Number of Layers: 2
- Dropout: 0.4
- Batch Size: 32
- Number of Epochs: 10
- Random Seed: 42 (for reproducibility)

## Data Information

**Dataset**: IMDB Movie Reviews

- **Training Samples**: 25,000
- **Test Samples**: 25,000
- **Task**: Binary sentiment classification (positive/negative)
- **Preprocessing**: Sequences truncated/padded to fixed length

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, sequence length, or hidden size |
| Package import errors | Rerun pip install cell at notebook start |
| Slow training | Ensure GPU is selected in Colab runtime settings |
| Results not saving | Check storage quota in Colab; download files before session ends |
