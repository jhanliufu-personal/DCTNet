import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import argparse
import random
import time
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, freqz
from scipy.stats import circmean

from ..models.DCTNN import DCTNN
from ..models.MLP import MLP, MLPComplex
from ..models.ECHTEstimator import ECHTEstimator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BaselineComparison:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        self._set_seed(self.config['seed'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use Agg backend for headless plotting if specified
        if self.config.get('headless_plotting', True):
            matplotlib.use('Agg')
        
        self.results = {}
    
    def _get_default_config(self):
        return {
            "test_data": {
                "x_test_path": "data/x_test_raw_nz04_6111595_20221222_8_theta.npy",
                "y_test_path": "data/y_test_filtered_nz04_6111595_20221222_8_theta.npy"
            },
            "models": {
                "dctnn": {
                    "real_model_path": "models/best_model_real.pth",
                    "imag_model_path": "models/best_model_imag.pth"
                },
                "mlp": {
                    "real_model_path": "models/best_mlp_real.pth",
                    "imag_model_path": "models/best_mlp_imag.pth"
                },
                "mlp_complex": {
                    "model_path": "models/best_mlp_complex.pth"
                }
            },
            "mlp_params": {
                "input_size": 512,
                "hidden_size": 256,
                "dropout_rate": 0.2
            },
            "echt_params": {
                "numerator": [0.010364, 0, -0.010364],
                "denominator": [1.0000, -1.9781, 0.9793],
                "sampling_rate": 1500
            },
            "normalization_factor": 6453,
            "batch_size": 64,
            "seed": 42,
            "visualization": {
                "start_idx": 30000,
                "end_idx": 32000,
                "sampling_rate": 1500
            },
            "headless_plotting": True
        }
    
    def _set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def freq_butterworth_filter(self, x):
        a = torch.tensor([[0.010364, 0, -0.010364]], dtype=torch.float32)
        b = torch.tensor([[1.0000, -1.9781, 0.9793]], dtype=torch.float32)
        
        A = torch.fft.fft(a, n=2048)
        B = torch.fft.fft(b, n=2048)
        
        H = A / B
        H = torch.abs(H).to(torch.float32)
        
        X = torch.fft.fft(x, n=2048)
        X = X * (H ** 2)
        
        x_filtered = torch.fft.ifft(X, n=2048).real
        return x_filtered[:, :512]
    
    def load_data(self):
        x_test = np.load(self.config['test_data']['x_test_path'])
        y_test = np.load(self.config['test_data']['y_test_path'])
        
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        x_test_tensor = self.freq_butterworth_filter(x_test_tensor)
        x_test_tensor = x_test_tensor / self.config['normalization_factor']
        
        y_test_real = torch.tensor(y_test[:, :1], dtype=torch.float32)
        y_test_imag = torch.tensor(y_test[:, 1:], dtype=torch.float32)
        
        return x_test_tensor, y_test_real, y_test_imag, x_test, y_test
    
    def calculate_metrics(self, phase_pred, phase_real, method_name):
        def rms_estimation_error(estimated_phases, ground_truth_phases):
            estm_error = np.abs(estimated_phases - ground_truth_phases)
            circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)
            return np.sqrt(np.mean(circular_estm_error**2))
        
        def mean_estimation_error(estimated_phases, ground_truth_phases):
            estm_error = np.abs(estimated_phases - ground_truth_phases)
            circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)
            return np.mean(circular_estm_error)
        
        phase_pred_flat = phase_pred.reshape(-1) if hasattr(phase_pred, 'reshape') else phase_pred.flatten()
        phase_real_flat = phase_real.reshape(-1) if hasattr(phase_real, 'reshape') else phase_real.flatten()
        
        rmse = rms_estimation_error(phase_pred_flat, phase_real_flat)
        mae = mean_estimation_error(phase_pred_flat, phase_real_flat)
        
        # Calculate circular mean error
        phase_diff = np.abs(phase_pred_flat - phase_real_flat)
        circular_mean_error = circmean(np.rad2deg(phase_diff), high=180, low=-180)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'circular_mean_error': circular_mean_error
        }
    
    def test_dctnn(self, x_test_tensor, y_test_real, y_test_imag):
        print("\\n=== Testing DCTNN ===")
        start_time = time.time()
        
        try:
            real_model = DCTNN().to(self.device)
            imag_model = DCTNN().to(self.device)
            
            real_model.load_state_dict(torch.load(
                self.config['models']['dctnn']['real_model_path'], 
                weights_only=False, map_location=self.device
            ))
            imag_model.load_state_dict(torch.load(
                self.config['models']['dctnn']['imag_model_path'], 
                weights_only=False, map_location=self.device
            ))
            
            real_model.eval()
            imag_model.eval()
            
            test_real_dataset = TensorDataset(x_test_tensor, y_test_real)
            test_imag_dataset = TensorDataset(x_test_tensor, y_test_imag)
            
            test_real_loader = DataLoader(test_real_dataset, batch_size=self.config['batch_size'], shuffle=False)
            test_imag_loader = DataLoader(test_imag_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            real_predictions = []
            imag_predictions = []
            
            with torch.no_grad():
                for (real_inputs, _), (imag_inputs, _) in zip(test_real_loader, test_imag_loader):
                    real_outputs = real_model(real_inputs.to(self.device))
                    imag_outputs = imag_model(imag_inputs.to(self.device))
                    
                    real_predictions.append(real_outputs.cpu().numpy())
                    imag_predictions.append(imag_outputs.cpu().numpy())
            
            real_predictions = np.concatenate(real_predictions, axis=0)
            imag_predictions = np.concatenate(imag_predictions, axis=0)
            
            phase_pred = np.arctan2(imag_predictions, real_predictions)
            
        except Exception as e:
            print(f"DCTNN test failed: {e}")
            return None, None, None, time.time() - start_time
        
        return real_predictions, imag_predictions, phase_pred, time.time() - start_time
    
    def test_mlp(self, x_test_tensor, y_test_real, y_test_imag):
        print("\\n=== Testing MLP ===")
        start_time = time.time()
        
        try:
            mlp_params = self.config['mlp_params']
            real_model = MLP(**mlp_params).to(self.device)
            imag_model = MLP(**mlp_params).to(self.device)
            
            real_model.load_state_dict(torch.load(
                self.config['models']['mlp']['real_model_path'], 
                weights_only=False, map_location=self.device
            ))
            imag_model.load_state_dict(torch.load(
                self.config['models']['mlp']['imag_model_path'], 
                weights_only=False, map_location=self.device
            ))
            
            real_model.eval()
            imag_model.eval()
            
            test_real_dataset = TensorDataset(x_test_tensor, y_test_real)
            test_imag_dataset = TensorDataset(x_test_tensor, y_test_imag)
            
            test_real_loader = DataLoader(test_real_dataset, batch_size=self.config['batch_size'], shuffle=False)
            test_imag_loader = DataLoader(test_imag_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            real_predictions = []
            imag_predictions = []
            
            with torch.no_grad():
                for (real_inputs, _), (imag_inputs, _) in zip(test_real_loader, test_imag_loader):
                    real_outputs = real_model(real_inputs.to(self.device))
                    imag_outputs = imag_model(imag_inputs.to(self.device))
                    
                    real_predictions.append(real_outputs.cpu().numpy())
                    imag_predictions.append(imag_outputs.cpu().numpy())
            
            real_predictions = np.concatenate(real_predictions, axis=0)
            imag_predictions = np.concatenate(imag_predictions, axis=0)
            
            phase_pred = np.arctan2(imag_predictions, real_predictions)
            
        except Exception as e:
            print(f"MLP test failed: {e}")
            return None, None, None, time.time() - start_time
        
        return real_predictions, imag_predictions, phase_pred, time.time() - start_time
    
    def test_mlp_complex(self, x_test_tensor, y_test):
        print("\\n=== Testing MLP Complex ===")
        start_time = time.time()
        
        try:
            mlp_params = self.config['mlp_params']
            model = MLPComplex(**mlp_params).to(self.device)
            
            model.load_state_dict(torch.load(
                self.config['models']['mlp_complex']['model_path'], 
                weights_only=False, map_location=self.device
            ))
            
            model.eval()
            
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            predictions = []
            
            with torch.no_grad():
                for inputs, _ in test_loader:
                    outputs = model(inputs.to(self.device))
                    predictions.append(outputs.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            real_predictions = predictions[:, :1]
            imag_predictions = predictions[:, 1:2]
            
            phase_pred = np.arctan2(imag_predictions, real_predictions)
            
        except Exception as e:
            print(f"MLP Complex test failed: {e}")
            return None, None, None, time.time() - start_time
        
        return real_predictions, imag_predictions, phase_pred, time.time() - start_time
    
    def test_echt(self, x_test):
        print("\\n=== Testing ecHT ===")
        start_time = time.time()
        
        try:
            echt_params = self.config['echt_params']
            estimator = ECHTEstimator(
                numerator=echt_params['numerator'],
                denominator=echt_params['denominator'], 
                fs=echt_params['sampling_rate']
            )
            
            num_samples, num_channels = x_test.shape
            analytic_signals = np.zeros((num_samples, num_channels), dtype=np.complex64)
            
            # Process each sample
            for i in range(num_samples):
                analytic_signal = estimator._echt(
                    x_test[i], 
                    echt_params['numerator'],
                    echt_params['denominator'],
                    echt_params['sampling_rate']
                )
                analytic_signals[i] = analytic_signal
            
            # Extract from last channel
            complex_last_channel = analytic_signals[:, -1]
            real_predictions = np.real(complex_last_channel).reshape(-1, 1)
            imag_predictions = np.imag(complex_last_channel).reshape(-1, 1)
            phase_pred = np.angle(complex_last_channel).reshape(-1, 1)
            
        except Exception as e:
            print(f"ecHT test failed: {e}")
            return None, None, None, time.time() - start_time
        
        return real_predictions, imag_predictions, phase_pred, time.time() - start_time
    
    def create_comparison_plots(self, results, y_test):
        print("\\n=== Creating Comparison Plots ===")
        
        phase_real = np.arctan2(y_test[:, 1:], y_test[:, :1])
        viz_config = self.config['visualization']
        start_idx = viz_config['start_idx']
        end_idx = viz_config['end_idx']
        sampling_rate = viz_config['sampling_rate']
        
        time_axis = np.arange(0, end_idx - start_idx) / sampling_rate
        
        # Phase comparison plot
        plt.figure(figsize=(15, 10))
        
        colors = ['gray', 'blue', 'red', 'green', 'orange']
        linestyles = ['-', '-', '--', '-.', ':']
        labels = ['Ground Truth']
        
        # Plot ground truth
        plt.plot(time_axis, phase_real[start_idx:end_idx].reshape(-1), 
                color=colors[0], label=labels[0], linewidth=2)
        
        # Plot predictions from each method
        method_idx = 1
        for method_name, result in results.items():
            if result['phase_pred'] is not None:
                plt.plot(time_axis, result['phase_pred'][start_idx:end_idx].reshape(-1),
                        color=colors[method_idx], linestyle=linestyles[method_idx],
                        label=f'{method_name.upper()}', linewidth=1.5, alpha=0.8)
                method_idx += 1
        
        plt.legend(fontsize=12)
        plt.title('Phase Estimation Comparison: All Methods', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Phase (radians)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('phase_comparison_all_methods.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual method plots
        for method_name, result in results.items():
            if result['phase_pred'] is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(time_axis, phase_real[start_idx:end_idx].reshape(-1), 
                        color='gray', label='Ground Truth', linewidth=2)
                plt.plot(time_axis, result['phase_pred'][start_idx:end_idx].reshape(-1),
                        color='blue', label=f'{method_name.upper()} Prediction', 
                        linewidth=1.5, alpha=0.8)
                plt.legend()
                plt.title(f'Phase Estimation: {method_name.upper()}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Phase (radians)')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'phase_comparison_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def run_comparison(self, methods=None):
        if methods is None:
            methods = ['dctnn', 'mlp', 'mlp_complex', 'echt']
        
        print("Starting baseline comparison...")
        print(f"Methods to test: {methods}")
        
        # Load test data
        print("\\nLoading test data...")
        x_test_tensor, y_test_real, y_test_imag, x_test, y_test = self.load_data()
        print(f"Test data shape: {x_test_tensor.shape}")
        
        # Ground truth phase
        phase_real = np.arctan2(y_test[:, 1:], y_test[:, :1])
        
        results = {}
        
        # Test each method
        if 'dctnn' in methods:
            real_pred, imag_pred, phase_pred, exec_time = self.test_dctnn(x_test_tensor, y_test_real, y_test_imag)
            if phase_pred is not None:
                metrics = self.calculate_metrics(phase_pred, phase_real, 'dctnn')
                results['dctnn'] = {
                    'real_predictions': real_pred,
                    'imag_predictions': imag_pred,
                    'phase_pred': phase_pred,
                    'metrics': metrics,
                    'execution_time': exec_time
                }
                print(f"DCTNN - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                      f"Circular Mean Error: {metrics['circular_mean_error']:.2f}°, Time: {exec_time:.2f}s")
        
        if 'mlp' in methods:
            real_pred, imag_pred, phase_pred, exec_time = self.test_mlp(x_test_tensor, y_test_real, y_test_imag)
            if phase_pred is not None:
                metrics = self.calculate_metrics(phase_pred, phase_real, 'mlp')
                results['mlp'] = {
                    'real_predictions': real_pred,
                    'imag_predictions': imag_pred,
                    'phase_pred': phase_pred,
                    'metrics': metrics,
                    'execution_time': exec_time
                }
                print(f"MLP - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                      f"Circular Mean Error: {metrics['circular_mean_error']:.2f}°, Time: {exec_time:.2f}s")
        
        if 'mlp_complex' in methods:
            real_pred, imag_pred, phase_pred, exec_time = self.test_mlp_complex(x_test_tensor, y_test)
            if phase_pred is not None:
                metrics = self.calculate_metrics(phase_pred, phase_real, 'mlp_complex')
                results['mlp_complex'] = {
                    'real_predictions': real_pred,
                    'imag_predictions': imag_pred,
                    'phase_pred': phase_pred,
                    'metrics': metrics,
                    'execution_time': exec_time
                }
                print(f"MLP Complex - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                      f"Circular Mean Error: {metrics['circular_mean_error']:.2f}°, Time: {exec_time:.2f}s")
        
        if 'echt' in methods:
            real_pred, imag_pred, phase_pred, exec_time = self.test_echt(x_test)
            if phase_pred is not None:
                metrics = self.calculate_metrics(phase_pred, phase_real, 'echt')
                results['echt'] = {
                    'real_predictions': real_pred,
                    'imag_predictions': imag_pred,
                    'phase_pred': phase_pred,
                    'metrics': metrics,
                    'execution_time': exec_time
                }
                print(f"ecHT - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                      f"Circular Mean Error: {metrics['circular_mean_error']:.2f}°, Time: {exec_time:.2f}s")
        
        # Create comparison plots
        if results:
            self.create_comparison_plots(results, y_test)
        
        # Print summary
        print("\\n" + "="*80)
        print("BASELINE COMPARISON SUMMARY")
        print("="*80)
        
        print(f"{'Method':<15} {'RMSE':<10} {'MAE':<10} {'Circular Error':<15} {'Time (s)':<10}")
        print("-" * 70)
        
        for method_name, result in results.items():
            if result['metrics']:
                metrics = result['metrics']
                print(f"{method_name.upper():<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
                      f"{metrics['circular_mean_error']:<15.2f} {result['execution_time']:<10.2f}")
        
        print("="*80)
        print("Plots saved:")
        print("- phase_comparison_all_methods.png")
        for method_name in results.keys():
            print(f"- phase_comparison_{method_name.lower()}.png")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparison for phase estimation.')
    parser.add_argument('--config', type=str, default='config/baseline_config.json',
                       help='Path to configuration file')
    parser.add_argument('--methods', nargs='+', 
                       choices=['dctnn', 'mlp', 'mlp_complex', 'echt'],
                       default=['dctnn', 'mlp', 'mlp_complex', 'echt'],
                       help='Methods to compare')
    parser.add_argument('--train-missing', action='store_true',
                       help='Train MLP models if they are missing')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        comparison = BaselineComparison()
    else:
        comparison = BaselineComparison(config_path=args.config)
    
    # Check if MLP models exist and train if requested
    if args.train_missing and ('mlp' in args.methods or 'mlp_complex' in args.methods):
        print("\\nChecking for missing MLP models...")
        missing_models = []
        
        if 'mlp' in args.methods:
            mlp_real_path = comparison.config['models']['mlp']['real_model_path']
            mlp_imag_path = comparison.config['models']['mlp']['imag_model_path']
            if not os.path.exists(mlp_real_path) or not os.path.exists(mlp_imag_path):
                missing_models.append('mlp')
        
        if 'mlp_complex' in args.methods:
            mlp_complex_path = comparison.config['models']['mlp_complex']['model_path']
            if not os.path.exists(mlp_complex_path):
                missing_models.append('mlp_complex')
        
        if missing_models:
            print(f"Missing models: {missing_models}")
            print("Please run: python train_baselines.py first")
            print("Or remove missing models from --methods argument")
            return
    
    # Run comparison
    results = comparison.run_comparison(methods=args.methods)
    
    print("\\nBaseline comparison done")


if __name__ == '__main__':
    main()