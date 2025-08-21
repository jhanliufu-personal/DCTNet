# DCTNet
Online phase estimation with dual-tree complex neural network with Discrete Cosine Transform (DCT) layers

---

## Quickstart

### 1. Environment Setup

Create and activate a conda environment:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate dctnet
```

Alternatively, if you prefer pip:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Or on Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Download and prepare the dataset:

```bash
# This will download, extract, and process all required data
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

## Data Preparation
Raw data can be downloaded [here](https://uchicago.box.com/s/pzrig7bi14ktjy9sl87si43jg3k7ukil).
Use scripts in the `Data/` to create the training and testing datasets.
- **`Extract_raw_data_from_pkl.py`**: Converts data from `.pkl` format to `.npy`.
- **`Generate_train_val_test.py`**: Splits the dataset into training, validation, and testing sets. This script supports three operational modes:
  - `Train_Val`: Generates only training and validation sets from the `.npy` file.
  - `Test`: Generates only the test set from the `.npy` file.
  - `Train_Val_Test`: Generates training, validation, and testing sets from the `.npy` file.

---

## Model Training
To train the model, execute the following command from the terminal:
```sh
python Train.py --training_model Both_Parts --data_dir path/to/data --save_dir save/outputs --epochs 1000 --batch_size 256 --data_fraction 0.01
```
Suggested Usage:

```sh
python Train.py --training_model Both_Parts --data_dir Data/Rodents_Raw_Data/ --save_dir ./ --epochs 1000 --batch_size 256 --data_fraction 1
```
This command runs `Train.py` with the specified parameters:
- Data used for training is the first 10 rodents from `sample_lfp_data_20240513.pkl`: [Download](https://uchicago.box.com/s/pzrig7bi14ktjy9sl87si43jg3k7ukil). It was devided into 80% training and 20% validation.

- `data_fraction`: Controls the percentage of data used for training.
- Training Modes:
  - `Real_Part`: Trains only the real part of the model.
  - `Imag_Part`: Trains only the imaginary part of the model.
  - `Both_Parts`: Trains the real part first, followed by the imaginary part.

---

## Model Evaluation
To ensure reproducibility, the `seed` and randomness settings have been predefined.

### Testing Methods:
You can perform testing using:
- The Python script: `Test.py`
- The Jupyter Notebook: `DCTNN Model Evaluation Notebook.ipynb`

### Dataset:
The test notebook utilizes rodent data available at:
- `x_test_raw_nz04_6111595_20221222_8_theta.npy`: [Download](https://uchicago.box.com/s/hvihloa9uc02ej9j5hfs1r5w07as6zj5)
- `y_test_filtered_nz04_6111595_20221222_8_theta.npy`: [Download](https://uchicago.box.com/s/noa156518fla2g05q1b8bnkw5blgbr1m)

### Evaluation Metrics:
The test code calculates the following performance metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Circular Estimation Box Plot
- Circular Mean Error (Model vs. Ground Truth and ecHT vs. Ground Truth)

### Generated Plots:
The test script also outputs:
- Predicted phase vs. ground truth
- Quantization performance
- ecHT phase vs. predicted phase vs. ground truth

For any issues or questions, please refer to the script comments or contact the author.

---
