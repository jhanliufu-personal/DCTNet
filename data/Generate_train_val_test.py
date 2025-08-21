import numpy as np
import os
import argparse
from scipy.signal import butter, sosfiltfilt, hilbert

def create_windows(data, window_size, step_size):
    return np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)]) 


def process_data_split(raw_signal, mode, sampling_rate=1500, window_size=512, 
                      step_size=1, lowcut=6, highcut=9, num_splits=6, 
                      split_all_factor=1.0, split_train_factor=0.8, 
                      output_dir='./processed_data'):
    
    os.makedirs(output_dir, exist_ok=True)
    butter_filter = butter(1, [lowcut, highcut], btype='bandpass', fs=sampling_rate, output='sos')
    epsilon = 1e-8
    
    split_size = len(raw_signal) // num_splits
    x_train_list, x_val_list, x_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []
    for i in range(num_splits):
        print(f"Processing part {i+1}/{num_splits}...")

        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else len(raw_signal)
        
        raw_chunk = raw_signal[start_idx:end_idx]
        filtered_chunk = sosfiltfilt(butter_filter, raw_chunk)
        analytical_signal_chunk = hilbert(filtered_chunk)

        if mode in ['Train_Val', 'Train_Val_Test']:
            split_index = int(len(raw_chunk) * split_all_factor)
            raw_signal_split = raw_chunk[:split_index]
            analytical_signal_split = analytical_signal_chunk[:split_index]

            raw_windows = create_windows(raw_signal_split, window_size, step_size)
            analytical_signal_windows = create_windows(analytical_signal_split, window_size, step_size)

            split_index_train = int(len(raw_windows) * split_train_factor)

            x_train_list.append(raw_windows[:split_index_train])
            x_val_list.append(raw_windows[split_index_train:])

            a_train = np.real(analytical_signal_windows[:split_index_train, -step_size:])
            b_train = np.imag(analytical_signal_windows[:split_index_train, -step_size:])
            a_val = np.real(analytical_signal_windows[split_index_train:, -step_size:])
            b_val = np.imag(analytical_signal_windows[split_index_train:, -step_size:])

            L_train = np.max(np.sqrt(a_train**2 + b_train**2)) + epsilon
            y_train_list.append(np.column_stack((a_train / L_train, b_train / L_train)).reshape(len(a_train), -1))

            L_val = np.max(np.sqrt(a_val**2 + b_val**2)) + epsilon
            y_val_list.append(np.column_stack((a_val / L_val, b_val / L_val)).reshape(len(a_val), -1))

        if mode in ['Test', 'Train_Val_Test']:
            filtered_resulttest_windows = create_windows(raw_chunk, window_size, step_size)
            analytical_signal_resulttest_windows = create_windows(analytical_signal_chunk, window_size, step_size)

            a_resulttest = np.real(analytical_signal_resulttest_windows[:, -step_size:])
            b_resulttest = np.imag(analytical_signal_resulttest_windows[:, -step_size:])

            L_resulttest = np.max(np.sqrt(a_resulttest**2 + b_resulttest**2)) + epsilon
            y_test_list.append(np.column_stack((a_resulttest / L_resulttest, b_resulttest / L_resulttest)).reshape(len(a_resulttest), -1))
            x_test_list.append(filtered_resulttest_windows)

    if mode in ['Train_Val', 'Train_Val_Test']:
        x_train = np.concatenate(x_train_list, axis=0)
        x_val = np.concatenate(x_val_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(output_dir, 'x_val.npy'), x_val)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        print(f"Training and validation data saved to {output_dir}")

    if mode in ['Test', 'Train_Val_Test']:
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        print(f"Test data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test datasets.')
    parser.add_argument('--input', default='Shuffled_10_Rodents_Raw_Data.npy', 
                       help='Input numpy file path')
    parser.add_argument('--mode', choices=['Train_Val', 'Test', 'Train_Val_Test'], 
                       default='Train_Val_Test', help='Processing mode')
    parser.add_argument('--output_dir', default='./processed_data', 
                       help='Output directory')
    parser.add_argument('--sampling_rate', type=int, default=1500, 
                       help='Sampling rate')
    parser.add_argument('--window_size', type=int, default=512, 
                       help='Window size')
    parser.add_argument('--step_size', type=int, default=1, 
                       help='Step size')
    parser.add_argument('--lowcut', type=float, default=6, 
                       help='Low cutoff frequency')
    parser.add_argument('--highcut', type=float, default=9, 
                       help='High cutoff frequency')
    parser.add_argument('--num_splits', type=int, default=6, 
                       help='Number of splits for memory management')
    parser.add_argument('--split_all_factor', type=float, default=1.0, 
                       help='Factor for train/val vs test split')
    parser.add_argument('--split_train_factor', type=float, default=0.8, 
                       help='Factor for train vs val split')
    
    args = parser.parse_args()
    
    raw_signal = np.load(args.input)
    print(f"Loaded raw signal with shape: {raw_signal.shape}")
    
    process_data_split(
        raw_signal=raw_signal,
        mode=args.mode,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        step_size=args.step_size,
        lowcut=args.lowcut,
        highcut=args.highcut,
        num_splits=args.num_splits,
        split_all_factor=args.split_all_factor,
        split_train_factor=args.split_train_factor,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
