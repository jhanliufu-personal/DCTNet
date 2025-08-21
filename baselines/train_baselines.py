import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ..models.MLP import MLP, MLPComplex


class BaselineTrainer:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)
    
    def _get_default_config(self):
        return {
            "training": {
                "data_dir": "./data",
                "save_dir": "./models",
                "epochs": 500,
                "batch_size": 512,
                "learning_rate": 0.001,
                "data_fraction": 1.0,
                "max_grad_norm": 1.0
            },
            "mlp_params": {
                "input_size": 512,
                "hidden_size": 256,
                "dropout_rate": 0.2
            }
        }
    
    def load_data(self):
        training_config = self.config['training']
        data_dir = training_config['data_dir']
        fraction = training_config['data_fraction']
        
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'), mmap_mode='r')
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'), mmap_mode='r')
        x_val = np.load(os.path.join(data_dir, 'x_val.npy'), mmap_mode='r')
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'), mmap_mode='r')
        
        num_samples_train = int(len(x_train) * fraction)
        num_samples_val = int(len(x_val) * fraction)
        
        x_train_tensor = torch.tensor(x_train[:num_samples_train], dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train[:num_samples_train], dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val[:num_samples_val], dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val[:num_samples_val], dtype=torch.float32)
        
        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor
    
    def create_data_loaders(self, x_train, y_train, x_val, y_val, complex_mode=False):
        batch_size = self.config['training']['batch_size']
        
        if complex_mode:
            # For complex MLP, use both real and imaginary parts together
            train_loader = DataLoader(
                TensorDataset(x_train, y_train), 
                batch_size=batch_size, 
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(x_val, y_val), 
                batch_size=batch_size, 
                shuffle=False
            )
            return train_loader, val_loader
        else:
            # Create separate data loaders for real and imaginary parts
            train_loader_real = DataLoader(
                TensorDataset(x_train, y_train[:, :1]), 
                batch_size=batch_size, 
                shuffle=True
            )
            val_loader_real = DataLoader(
                TensorDataset(x_val, y_val[:, :1]), 
                batch_size=batch_size, 
                shuffle=False
            )
            train_loader_imag = DataLoader(
                TensorDataset(x_train, y_train[:, 1:]), 
                batch_size=batch_size, 
                shuffle=True
            )
            val_loader_imag = DataLoader(
                TensorDataset(x_val, y_val[:, 1:]), 
                batch_size=batch_size, 
                shuffle=False
            )
            
            return train_loader_real, val_loader_real, train_loader_imag, val_loader_imag
    
    def train_model(self, model, train_loader, val_loader, save_path, part_name):
        training_config = self.config['training']
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"Training {part_name}...")
        
        for epoch in range(training_config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             max_norm=training_config.get('max_grad_norm', 1.0))
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{training_config["epochs"]}, '
                      f'Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, save_path)
                if (epoch + 1) % 50 == 0:
                    print(f'Best {part_name} model saved at epoch {epoch+1}')
        
        print(f'{part_name} training completed. Best validation loss: {best_val_loss:.6f}')
        return best_val_loss
    
    def train_mlp_separate(self):
        print("\\n=== Training Separate MLP Models ===")
        
        # Load data
        print("Loading data...")
        x_train, y_train, x_val, y_val = self.load_data()
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {x_val.shape}, {y_val.shape}")
        
        # Create data loaders
        train_loader_real, val_loader_real, train_loader_imag, val_loader_imag = \
            self.create_data_loaders(x_train, y_train, x_val, y_val, complex_mode=False)
        
        mlp_params = self.config['mlp_params']
        results = {}
        
        # Train real part MLP
        print("\\n--- Training MLP for Real Part ---")
        real_model = MLP(**mlp_params).to(self.device)
        real_save_path = os.path.join(self.config['training']['save_dir'], 'best_mlp_real.pth')
        real_loss = self.train_model(real_model, train_loader_real, val_loader_real, real_save_path, "MLP Real")
        results['mlp_real_best_loss'] = real_loss
        
        # Train imaginary part MLP
        print("\\n--- Training MLP for Imaginary Part ---")
        imag_model = MLP(**mlp_params).to(self.device)
        imag_save_path = os.path.join(self.config['training']['save_dir'], 'best_mlp_imag.pth')
        imag_loss = self.train_model(imag_model, train_loader_imag, val_loader_imag, imag_save_path, "MLP Imaginary")
        results['mlp_imag_best_loss'] = imag_loss
        
        return results
    
    def train_mlp_complex(self):
        print("\\n=== Training Complex MLP Model ===")
        
        # Load data
        print("Loading data...")
        x_train, y_train, x_val, y_val = self.load_data()
        
        # Create data loaders for complex training
        train_loader, val_loader = self.create_data_loaders(x_train, y_train, x_val, y_val, complex_mode=True)
        
        # Train complex MLP
        mlp_params = self.config['mlp_params']
        complex_model = MLPComplex(**mlp_params).to(self.device)
        complex_save_path = os.path.join(self.config['training']['save_dir'], 'best_mlp_complex.pth')
        complex_loss = self.train_model(complex_model, train_loader, val_loader, complex_save_path, "MLP Complex")
        
        return {'mlp_complex_best_loss': complex_loss}
    
    def train_all(self):
        print("Starting baseline model training...")
        
        all_results = {}
        
        # Train separate MLP models
        mlp_results = self.train_mlp_separate()
        all_results.update(mlp_results)
        
        # Train complex MLP model
        complex_results = self.train_mlp_complex()
        all_results.update(complex_results)
        
        print("\\n" + "="*50)
        print("BASELINE TRAINING SUMMARY")
        print("="*50)
        print("Results:")
        for key, value in all_results.items():
            print(f"  {key}: {value:.6f}")
        
        # Save training configuration
        config_save_path = os.path.join(self.config['training']['save_dir'], 'baseline_training_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\\nTraining configuration saved to: {config_save_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Train baseline MLP models.')
    parser.add_argument('--config', type=str, default='config/baseline_config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', choices=['mlp', 'mlp_complex', 'all'], default='all',
                       help='Which model to train')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./models', 
                       help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, 
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create trainer with config file or command line arguments
    if args.config and os.path.exists(args.config):
        trainer = BaselineTrainer(config_path=args.config)
    else:
        print("Config file not found, using command line arguments...")
        # Create config from command line arguments
        config = {
            "training": {
                "data_dir": args.data_dir,
                "save_dir": args.save_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "data_fraction": 1.0,
                "max_grad_norm": 1.0
            },
            "mlp_params": {
                "input_size": 512,
                "hidden_size": 256,
                "dropout_rate": 0.2
            }
        }
        
        # Save temporary config file
        temp_config_path = os.path.join(args.save_dir, 'temp_baseline_config.json')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        trainer = BaselineTrainer(config_path=temp_config_path)
    
    # Run training based on selected model
    if args.model == 'mlp':
        results = trainer.train_mlp_separate()
    elif args.model == 'mlp_complex':
        results = trainer.train_mlp_complex()
    else:  # 'all'
        results = trainer.train_all()
    
    print("\\nBaseline training completed!")
    print("\\nYou can now run baseline comparison with:")
    print("  python run_baselines.py")


if __name__ == '__main__':
    main()