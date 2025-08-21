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
- Verify all data files are correctly processed

### 3. Model Training

Train the DCTNN model:

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

### 5. Configuration

You can customize training and testing parameters by editing the configuration files in the `config/` directory:

- `config/train_config.json` - Training parameters
- `config/test_config.json` - Testing parameters  
- `config/data_prep_config.json` - Data preparation settings

---

## Model Evaluation
We evaluate the model and baseline methods using the following performance metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Circular Estimation Box Plot
- Circular Mean Error (Model vs. Ground Truth and ecHT vs. Ground Truth)

---
