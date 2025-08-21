import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import argparse
import random
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, freqz
from scipy.stats import circmean
from models.model_POSE_DCTNN9 import DCTNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DCTNNTester:
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
    
    def _get_default_config(self):
        return {
            "test_data": {
                "x_test_path": "Data/x_test_raw_nz04_6111595_20221222_8_theta.npy",
                "y_test_path": "Data/y_test_filtered_nz04_6111595_20221222_8_theta.npy"
            },
            "models": {
                "real_model_path": "Trained Models/Final DCTNN Model trained on 11 Rodents for Freq-doamin Butterworth Filtering/best_model_real_11Rats_Norm2.pth",
                "imag_model_path": "Trained Models/Final DCTNN Model trained on 11 Rodents for Freq-doamin Butterworth Filtering/best_model_imag_11Rats_Norm2.pth"
            },
            "normalization_factor": 6453,
            "batch_size": 64,
            "seed": 42,
            "visualization": {
                "start_idx": 30000,
                "end_idx": 32000,
                "sampling_rate": 1500
            },
            "echt_params": {
                "filt_lf": 6,
                "filt_hf": 9,
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
    
    def load_models(self):
        real_model = DCTNN().to(self.device)
        imag_model = DCTNN().to(self.device)
        
        real_model.load_state_dict(torch.load(
            self.config['models']['real_model_path'], 
            weights_only=False,
            map_location=self.device
        ))
        imag_model.load_state_dict(torch.load(
            self.config['models']['imag_model_path'], 
            weights_only=False,
            map_location=self.device
        ))
        
        real_model.eval()
        imag_model.eval()
        
        return real_model, imag_model
    
    def predict(self, real_model, imag_model, x_test_tensor, y_test_real, y_test_imag):
        test_real_dataset = TensorDataset(x_test_tensor, y_test_real)
        test_imag_dataset = TensorDataset(x_test_tensor, y_test_imag)
        
        test_real_loader = DataLoader(test_real_dataset, 
                                    batch_size=self.config['batch_size'], 
                                    shuffle=False)
        test_imag_loader = DataLoader(test_imag_dataset, 
                                    batch_size=self.config['batch_size'], 
                                    shuffle=False)
        
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
        
        return real_predictions, imag_predictions
    
    def calculate_metrics(self, phase_pred, phase_real):
        def rms_estimation_error(estimated_phases, ground_truth_phases):
            estm_error = np.abs(estimated_phases - ground_truth_phases)
            circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)
            return np.sqrt(np.mean(circular_estm_error**2))
        
        def mean_estimation_error(estimated_phases, ground_truth_phases):
            estm_error = np.abs(estimated_phases - ground_truth_phases)
            circular_estm_error = np.where(estm_error > np.pi, 2*np.pi - estm_error, estm_error)
            return np.mean(circular_estm_error)
        
        rmse = rms_estimation_error(phase_pred.reshape(-1), phase_real.reshape(-1))
        mae = mean_estimation_error(phase_pred.reshape(-1), phase_real.reshape(-1))
        
        return rmse, mae
    
    def echt(self, xr, filt_lf, filt_hf, Fs):
        n = xr.shape[0]
        X = np.fft.fft(xr, n=n)
        
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        
        X = X * h
        
        filt_order = 2
        b, a = butter(filt_order, [filt_lf / (Fs / 2), filt_hf / (Fs / 2)], btype='bandpass')
        
        filt_freq = np.fft.fftfreq(n, d=1/Fs)
        filt_freq = np.fft.fftshift(filt_freq)
        _, filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)
        
        X = np.fft.fftshift(X)
        X = X * filt_coeff
        X = np.fft.ifftshift(X)
        
        return np.fft.ifft(X)
    
    def create_visualizations(self, phase_pred, phase_real, real_predictions, imag_predictions,
                             y_test_real, y_test_imag, x_test):
        viz_config = self.config['visualization']
        start_idx = viz_config['start_idx']
        end_idx = viz_config['end_idx']
        sampling_rate = viz_config['sampling_rate']
        
        time_axis = np.arange(0, end_idx - start_idx) / sampling_rate
        
        # Phase comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, phase_real[start_idx:end_idx].reshape(-1), 
                color='gray', label='True Phase')
        plt.plot(time_axis, phase_pred[start_idx:end_idx].reshape(-1), 
                color='blue', label='Predicted Phase')
        plt.legend()
        plt.title('True Phase vs Predicted Phase')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Phase')
        plt.savefig('Phase.png')
        plt.close()
        
        # Real part comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, y_test_real[start_idx:end_idx].reshape(-1), 
                color='gray', label='True Real Part')
        plt.plot(time_axis, real_predictions[start_idx:end_idx].reshape(-1), 
                color='blue', label='Predicted Real Part')
        plt.legend()
        plt.title('True Real Part vs Predicted Real Part')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Real Part')
        plt.savefig('Real_Part.png')
        plt.close()
        
        # Imaginary part comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, y_test_imag[start_idx:end_idx].reshape(-1), 
                color='gray', label='True Imaginary Part')
        plt.plot(time_axis, imag_predictions[start_idx:end_idx].reshape(-1), 
                color='blue', label='Predicted Imaginary Part')
        plt.legend()
        plt.title('True Imaginary Part vs Predicted Imaginary Part')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Imaginary Part')
        plt.savefig('Imaginary_Part.png')
        plt.close()
        
        # Quantization plot
        plt.figure()
        plt.scatter(phase_real.reshape(-1), phase_pred.reshape(-1), s=0.001)
        plt.xlabel('Ground truth phase')
        plt.ylabel('Predicted phase')
        diagonal = np.arange(-np.pi, np.pi)
        plt.plot(diagonal, diagonal, ls='--', c='r')
        plt.savefig('Quantization.png')
        plt.axis('square')
        plt.close()
        
        # Circular estimation error boxplot
        def circ_diff(unit1, unit2):
            phi = np.remainder(unit2 - unit1, 2 * np.pi)
            return np.where(phi > np.pi, phi - 2 * np.pi, phi)
        
        circular_err = circ_diff(phase_pred.reshape(-1), phase_real.reshape(-1))
        num_bins = 10
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        bin_indices = np.digitize(phase_real.reshape(-1), bins)
        binned_errors = [circular_err[bin_indices == i] for i in range(1, len(bins))]
        
        plt.figure()
        plt.boxplot(binned_errors, 
                   positions=(bins[:-1] + bins[1:]) / 2,
                   widths=np.pi / num_bins,
                   showfliers=False,
                   medianprops={'linewidth': 2.5})
        plt.axhline(y=0, ls='--', c='k', alpha=0.3)
        plt.xlabel("Ground truth phase")
        plt.ylabel("Circular error")
        plt.title("Circular estimation errors binned by ground truth phase")
        plt.xticks(ticks=(bins[:-1] + bins[1:]) / 2,
                  labels=[f"{round(x, 2)}" for x in (bins[:-1] + bins[1:]) / 2],
                  rotation=25)
        plt.tight_layout()
        plt.ylim(-np.pi, np.pi)
        plt.savefig('Circular_estimation_errors.png')
        plt.axis('square')
        plt.close()
    
    def compare_with_echt(self, x_test, real_predictions, imag_predictions, y_test_real, y_test_imag):
        echt_params = self.config['echt_params']
        num_samples, num_channels = x_test.shape
        
        analytic_signal_echt = np.zeros((num_samples, num_channels), dtype=np.complex64)
        for i in range(num_samples):
            analytic_signal_echt[i] = self.echt(x_test[i], 
                                              echt_params['filt_lf'],
                                              echt_params['filt_hf'], 
                                              echt_params['sampling_rate'])
        
        phase_echt = np.angle(analytic_signal_echt[:, -1])
        
        # Calculate DCTNN circular mean error
        analytic_signal = y_test_real + 1j * y_test_imag
        phase_real_analytic = np.angle(analytic_signal)
        
        analytic_pred = real_predictions + 1j * imag_predictions
        phase_pred_analytic = np.angle(analytic_pred)
        
        phase_diff_dctnn = np.abs(phase_real_analytic - phase_pred_analytic)
        circular_mean_dctnn = circmean(np.rad2deg(phase_diff_dctnn), high=180, low=-180)
        print(f"Average Circular Mean DCTNN error: {circular_mean_dctnn} degrees")
        
        # Calculate ecHT circular mean error
        phase_diff_echt = np.abs(phase_real_analytic.squeeze() - phase_echt)
        circular_mean_echt = circmean(np.rad2deg(phase_diff_echt), high=180, low=-180)
        print(f"Average Circular Mean ecHT error: {circular_mean_echt} degrees")
        
        # Create comparison plot
        viz_config = self.config['visualization']
        start_idx = viz_config['start_idx']
        end_idx = viz_config['end_idx']
        sampling_rate = viz_config['sampling_rate']
        
        time_axis = np.arange(0, end_idx - start_idx) / sampling_rate
        
        plt.figure(figsize=(14, 7))
        plt.plot(time_axis, phase_echt[start_idx:end_idx], 
                label='ecHT Phase', color='blue', linewidth=2)
        plt.plot(time_axis, phase_pred_analytic[start_idx:end_idx], 
                label='Predicted Phase', color='red', linestyle='dashed', linewidth=2, alpha=0.8)
        plt.plot(time_axis, phase_real_analytic[start_idx:end_idx], 
                label='Original Signal', color='green', linestyle='dashdot', linewidth=1.5, alpha=0.7)
        
        plt.legend(fontsize=12, loc='upper right')
        plt.title('Comparison of ecHT Phase, Predicted Phase, and Original Signal', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Phase (radians)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('echt_vs_predicted_vs_original_phase.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return circular_mean_dctnn, circular_mean_echt
    
    def run_test(self):
        print("Loading data...")
        x_test_tensor, y_test_real, y_test_imag, x_test, y_test = self.load_data()
        print(f"x_test shape: {x_test_tensor.shape}")
        
        print("Loading models...")
        real_model, imag_model = self.load_models()
        
        print("Making predictions...")
        real_predictions, imag_predictions = self.predict(
            real_model, imag_model, x_test_tensor, y_test_real, y_test_imag
        )
        
        print("Calculating phase...")
        phase_pred = np.arctan2(imag_predictions, real_predictions)
        phase_real = np.arctan2(y_test[:, 1:], y_test[:, :1])
        
        print("Calculating metrics...")
        rmse, mae = self.calculate_metrics(phase_pred, phase_real)
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        
        print("Creating visualizations...")
        self.create_visualizations(phase_pred, phase_real, real_predictions, 
                                 imag_predictions, y_test[:, :1], y_test[:, 1:], x_test)
        
        print("Comparing with ecHT...")
        dctnn_error, echt_error = self.compare_with_echt(
            x_test, real_predictions, imag_predictions, y_test[:, :1], y_test[:, 1:]
        )
        
        print("Testing completed successfully!")
        return {
            'rmse': rmse,
            'mae': mae,
            'dctnn_circular_mean_error': dctnn_error,
            'echt_circular_mean_error': echt_error
        }


def main():
    parser = argparse.ArgumentParser(description='Test DCTNN model.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    tester = DCTNNTester(config_path=args.config)
    results = tester.run_test()
    
    print("\nTest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    main()