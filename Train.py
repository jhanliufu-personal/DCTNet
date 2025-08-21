import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.DCTNN import DCTNN


class DCTNNTrainer:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
    
    def _get_default_config(self):
        return {
            "data_dir": "./data",
            "save_dir": "./models",
            "training_model": "Both_Parts",
            "epochs": 1500,
            "batch_size": 512,
            "data_fraction": 1.0,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0
        }
    
    def load_data(self):
        data_dir = self.config['data_dir']
        fraction = self.config['data_fraction']
        
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
    
    def create_data_loaders(self, x_train, y_train, x_val, y_val):
        batch_size = self.config['batch_size']
        
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
    
    def train_model_part(self, model, train_loader, val_loader, save_path, part_name):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"Training {part_name} part...")
        
        for epoch in range(self.config['epochs']):
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
                                             max_norm=self.config['max_grad_norm'])
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
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}, '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, save_path)
                print(f'Best {part_name} model saved at epoch {epoch+1}')
        
        print(f'{part_name} part training completed. Best validation loss: {best_val_loss:.6f}')
        return best_val_loss
    
    def train(self):
        print("Loading data...")
        x_train, y_train, x_val, y_val = self.load_data()
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {x_val.shape}, {y_val.shape}")
        
        print("Creating data loaders...")
        train_loader_real, val_loader_real, train_loader_imag, val_loader_imag = \
            self.create_data_loaders(x_train, y_train, x_val, y_val)
        
        training_mode = self.config['training_model']
        results = {}
        
        # Initialize models
        model_real = DCTNN().to(self.device) if training_mode in ['Real_Part', 'Both_Parts'] else None
        model_imag = DCTNN().to(self.device) if training_mode in ['Imag_Part', 'Both_Parts'] else None
        
        # Train real part
        if training_mode in ['Real_Part', 'Both_Parts']:
            real_save_path = os.path.join(self.config['save_dir'], 'best_model_real.pth')
            real_loss = self.train_model_part(
                model_real, train_loader_real, val_loader_real, real_save_path, "Real"
            )
            results['real_part_best_loss'] = real_loss
        
        # Train imaginary part
        if training_mode in ['Imag_Part', 'Both_Parts']:
            imag_save_path = os.path.join(self.config['save_dir'], 'best_model_imag.pth')
            imag_loss = self.train_model_part(
                model_imag, train_loader_imag, val_loader_imag, imag_save_path, "Imaginary"
            )
            results['imag_part_best_loss'] = imag_loss
        
        print("\nTraining completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        return results
    
    def save_config(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.config['save_dir'], 'training_config.json')
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Training configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train DCTNN model.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--training_model', type=str, default='Both_Parts', 
                       choices=['Real_Part', 'Imag_Part', 'Both_Parts'],
                       help='Training mode')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./models', 
                       help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=1500, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, 
                       help='Batch size')
    parser.add_argument('--data_fraction', type=float, default=1.0, 
                       help='Fraction of data to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create trainer with config file or command line arguments
    if args.config:
        trainer = DCTNNTrainer(config_path=args.config)
    else:
        # Create config from command line arguments
        config = {
            "training_model": args.training_model,
            "data_dir": args.data_dir,
            "save_dir": args.save_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "data_fraction": args.data_fraction,
            "learning_rate": args.learning_rate,
            "max_grad_norm": 1.0
        }
        
        # Save temporary config file
        temp_config_path = os.path.join(args.save_dir, 'temp_config.json')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        trainer = DCTNNTrainer(config_path=temp_config_path)
    
    # Run training
    results = trainer.train()
    
    # Save final configuration
    trainer.save_config()


if __name__ == '__main__':
    main()