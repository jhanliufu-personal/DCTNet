# DCTNet
Online phase estimation with dual-tree complex neural network with Discrete Cosine Transform (DCT) layers

---

## Quickstart

### 1. Environment Setup

Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate dctnet
```

Alternatively, if you prefer pip:

```bash
python -m venv dctnet
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation

Download and prepare the dataset:

```bash
python prepare_data.py
```

The script will:
- Guide you through downloading the required data files
- Extract raw data from pickle files
- Generate train/validation/test datasets

### 3. Model Training

Train the DCTNN model using the following commands. We recommend training the real and imaginary
part together (`Both_Parts`).

```bash
# Train both real and imaginary parts with default settings
python Train.py --config config/train_config.json

# Or train with custom parameters
python Train.py --training_model Both_Parts --data_dir data --save_dir models --epochs 1000 --batch_size 256
```

### 4. Model Testing

Test the trained model:

```bash
# Test with default configuration
python Test.py --config config/test_config.json

# This will generate evaluation metrics and visualization plots
```

### 5. Baseline Comparison

To compare DCTNN against baseline methods, first train the baseline models:

```bash
# Train MLP baseline models (both separate and complex versions)
python train_baselines.py --config config/baseline_config.json

# Or train with custom parameters
python train_baselines.py --epochs 500 --batch_size 512 --learning_rate 0.001
```

Then run the baseline comparison:

```bash
# Compare all methods (DCTNN, MLP, MLP Complex, ecHT)
python run_baselines.py --config config/baseline_config.json

# Compare specific methods only
python run_baselines.py --methods dctnn mlp echt

# Train missing models automatically
python run_baselines.py --train-missing
```

This will generate:
- Comparison metrics (RMSE, MAE, Circular Mean Error)
- Phase estimation plots for all methods
- Individual method comparison plots

### 6. Configuration

You can customize training and testing parameters by editing the configuration files in the `config/` directory:

- `config/train_config.json` - Training parameters
- `config/test_config.json` - Testing parameters  
- `config/data_prep_config.json` - Data preparation settings
- `config/baseline_config.json` - Baseline comparison parameters

---

## Model Evaluation
We evaluate the model and baseline methods using the following performance metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Circular Estimation Box Plot
- Circular Mean Error (Model vs. Ground Truth and ecHT vs. Ground Truth)

---
